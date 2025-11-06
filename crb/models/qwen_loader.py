# src/crb/models/qwen_loader.py
from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class QwenLoadConfig:
    model_id: str
    peft_model_id: Optional[str] = None
    attn_implementation: str = "flash_attention_2"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    local_files_only: bool = True
    trust_remote_code: bool = True

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

def load_qwen(cfg: QwenLoadConfig):
    model_path = cfg.peft_model_id or cfg.model_id
    print(f"🚀 正在加载模型: {model_path}")
    torch_dtype = _DTYPE_MAP.get(cfg.dtype, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation=cfg.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=cfg.device_map
    )

    print(f"🚀 正在加载 Tokenizer: {cfg.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id,
        local_files_only=cfg.local_files_only,
        trust_remote_code=cfg.trust_remote_code
    )
    # eos / pad 设置（与你给的示意一致）
    try:
        eos_token_id = tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"][0]
        tokenizer.eos_token_id = eos_token_id
    except Exception:
        eos_token_id = tokenizer.eos_token_id

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    torch.set_grad_enabled(False)
    print("✅ 模型和 Tokenizer 加载完毕")
    return model, tokenizer
