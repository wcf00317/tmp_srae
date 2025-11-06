# src/crb/pipelines/collect_activations.py
from __future__ import annotations
import os
import json
import argparse
import numpy as np
import torch
from typing import List, Dict, Any, Tuple

from transformers import PreTrainedTokenizer
from crb.models.qwen_loader import load_qwen, QwenLoadConfig
from crb.data.gsm8k import iter_gsm8k, extract_final_number, equals_num
from crb.instrumentation.layer_selectors import pick_blocks_by_indices
from crb.instrumentation.activation_recorder import ActivationRecorder
from crb.instrumentation.reasoning_mask import compute_reasoning_mask

def build_messages(question: str, system: str, user_suffix: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question.strip() + (user_suffix or "")},
    ]

def encode_chat(tokenizer: PreTrainedTokenizer, messages: List[Dict[str,str]]) -> torch.Tensor:
    """使用 Qwen/chat 模板编码为 input_ids（batch=1）"""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    return ids

@torch.inference_mode()
def step_decode_collect(model, tokenizer, input_ids: torch.Tensor, blocks, layers_idx: List[int],
                        max_new_tokens: int, temperature: float, top_p: float, stop_on_eos: bool) -> Dict[str, Any]:
    """
    增量解码并在每步收集指定层的隐状态。
    返回：
      gen_ids: List[int]           # 生成序列（不含 prompt）
      acts: np.ndarray[T, L, D]
      text: str                    # 生成文本
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    eos_id = tokenizer.eos_token_id

    # 安装 hook
    recorder = ActivationRecorder(blocks, capture="output", dtype=torch.bfloat16)

    # 先一次性跑 prompt，获取初始 past
    out = model(input_ids=input_ids, use_cache=True)
    past_kv = out.past_key_values

    gen_ids: List[int] = []
    feats: List[torch.Tensor] = []

    # 逐步生成
    cur = input_ids[:, -1:]
    for step in range(max_new_tokens):
        recorder.start_step()
        logits = model(input_ids=cur, use_cache=True, past_key_values=past_kv).logits
        past_kv = model.past_key_values  # 有些实现把 past 作为内部状态暴露；稳妥起见也从输出取
        if past_kv is None and hasattr(out, "past_key_values"):
            past_kv = out.past_key_values

        next_token_logits = logits[:, -1, :]  # (1, V)
        if temperature > 0.0:
            probs = torch.softmax(next_token_logits / max(1e-6, temperature), dim=-1)
            if top_p < 1.0:
                # nucleus sampling
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                mask = cum <= top_p
                mask[..., 0] = True
                idx_keep = sorted_idx[mask]
                probs_keep = probs.gather(-1, idx_keep.unsqueeze(0))
                probs_keep = probs_keep / probs_keep.sum(dim=-1, keepdim=True)
                next_token = idx_keep[torch.multinomial(probs_keep, num_samples=1)]
            else:
                next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # 贪婪

        # 取本步捕获到的 [L, D]
        step_feats = recorder.pop_step_features()  # [L, D]
        feats.append(step_feats)
        tok = int(next_token.item())
        gen_ids.append(tok)

        if stop_on_eos and tok == eos_id:
            break

        cur = next_token

    recorder.remove()

    acts = torch.stack(feats, dim=0).cpu().numpy().astype(np.float16)  # [T, L, D]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {
        "gen_ids": gen_ids,
        "acts": acts,
        "text": gen_text,
    }

def save_npz(sample_dir: str, qid: str, payload: Dict[str, Any]):
    os.makedirs(sample_dir, exist_ok=True)
    path = os.path.join(sample_dir, f"{qid}.npz")
    np.savez_compressed(path, **payload)
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/collect_acts.yaml", help="YAML 配置路径")
    args = ap.parse_args()

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1) 加载模型/分词器
    mdl_cfg = QwenLoadConfig(
        model_id=cfg["model"]["model_id"],
        peft_model_id=cfg["model"].get("peft_model_id") or cfg["model"]["model_id"],
        attn_implementation=cfg["model"].get("attn_implementation", "flash_attention_2"),
        dtype=cfg["model"].get("dtype", "bfloat16"),
        device_map=cfg["model"].get("device_map", "auto"),
        local_files_only=cfg["model"].get("local_files_only", True),
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
    )
    model, tokenizer = load_qwen(mdl_cfg)

    # 2) 选层并装好 blocks
    layers_idx: List[int] = cfg["collect"]["layers"]
    from crb.instrumentation.layer_selectors import pick_blocks_by_indices
    blocks, block_names = pick_blocks_by_indices(model, layers_idx)
    print(f"📌 采集层: {block_names}")

    # 3) 数据
    limit = cfg["data"].get("limit")
    giter = iter_gsm8k(cfg["data"]["split"], limit=limit)

    # 4) 输出
    out_dir = cfg["collect"]["save_dir"]
    os.makedirs(out_dir, exist_ok=True)
    write_txt = bool(cfg["io"].get("write_text_preview", True))

    # 5) 解码参数
    max_new_tokens = int(cfg["collect"]["max_new_tokens"])
    temperature = float(cfg["collect"].get("temperature", 0.0))
    top_p = float(cfg["collect"].get("top_p", 1.0))
    stop_on_eos = bool(cfg["collect"].get("stop_on_eos", True))

    # 6) 提示模板
    system = cfg["prompt"]["system"]
    user_suffix = cfg["prompt"].get("user_suffix", "")

    # 7) 主循环
    meta_index = []
    for k, ex in enumerate(giter, start=1):
        qid = str(ex["id"])
        question = ex["question"]
        gold_norm = ex["gold_norm"]

        messages = build_messages(question, system, user_suffix)
        input_ids = encode_chat(tokenizer, messages)

        with torch.no_grad():
            result = step_decode_collect(
                model, tokenizer, input_ids, blocks, layers_idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, stop_on_eos=stop_on_eos
            )

        gen_ids = result["gen_ids"]
        gen_text = result["text"]
        acts = result["acts"]  # [T, L, D]

        # 生成级指标
        pred_norm = extract_final_number(gen_text)
        success = equals_num(pred_norm, gold_norm)

        # 推理掩码（针对生成 tokens）
        reasoning_mask = compute_reasoning_mask(gen_ids, tokenizer, window=0)

        payload = {
            "question_id": qid,
            "gen_ids": np.array(gen_ids, dtype=np.int32),
            "acts": acts,                                  # [T, L, D] fp16
            "layers": np.array(layers_idx, dtype=np.int32),
            "reasoning_mask": np.array(reasoning_mask, dtype=bool),
            "gold": gold_norm if gold_norm is not None else "",
            "pred": pred_norm if pred_norm is not None else "",
            "success_flag": bool(success),
        }
        path = save_npz(out_dir, qid, payload)

        if write_txt:
            with open(os.path.join(out_dir, f"{qid}.txt"), "w", encoding="utf-8") as fw:
                fw.write(f"# id: {qid}\n\n")
                fw.write("## Question\n")
                fw.write(question.strip() + "\n\n")
                fw.write("## Generated\n")
                fw.write(gen_text.strip() + "\n\n")
                fw.write(f"## Gold (norm): {gold_norm}\n")
                fw.write(f"## Pred (norm): {pred_norm}\n")
                fw.write(f"## Success: {success}\n")

        meta_index.append({
            "id": qid,
            "path": os.path.abspath(path),
            "T": int(acts.shape[0]),
            "L": int(acts.shape[1]),
            "D": int(acts.shape[2]),
            "success": bool(success),
        })

        if k % 10 == 0:
            print(f"✅ 已完成 {k} 条，最后一条保存于: {path}")

    # 写一个简单索引
    with open(os.path.join(out_dir, "_index.jsonl"), "w", encoding="utf-8") as fw:
        for row in meta_index:
            fw.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"🎉 全部完成，共 {len(meta_index)} 条。索引: {os.path.join(out_dir, '_index.jsonl')}")


if __name__ == "__main__":
    main()
