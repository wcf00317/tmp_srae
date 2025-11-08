#!/usr/bin/env python3
# scripts/prepare_rsae_corpus.py
from __future__ import annotations
import os, json, argparse
from pathlib import Path
import numpy as np

def dilate_mask(mask: np.ndarray, w: int) -> np.ndarray:
    if w <= 0: return mask
    n = len(mask)
    out = np.zeros_like(mask, dtype=bool)
    idx = np.where(mask)[0]
    for i in idx:
        lo = max(0, i-w); hi = min(n, i+w+1)
        out[lo:hi] = True
    return out

def iter_npz_files(root: Path):
    for p in sorted(root.rglob("*.npz")):
        yield p

def per_layer_centering(acts_tld: np.ndarray) -> np.ndarray:
    # acts_tld: [T, L, D], 返回逐层逐题中心化后的同形状数组
    # 对每个层 l：x'_{t,l} = x_{t,l} - mean_t(x_{t,l})
    T, L, D = acts_tld.shape
    out = acts_tld.astype(np.float32, copy=True)
    mean_t = out.mean(axis=0, keepdims=True)  # [1, L, D]
    out -= mean_t
    return out

def compute_rms(x: np.ndarray, axis: int = 0, eps: float = 1e-6):
    # x: [N, D]，返回 [D] 的 rms
    return np.sqrt((x.astype(np.float32) ** 2).mean(axis=axis) + eps)

def main():
    ap = argparse.ArgumentParser(description="Prepare reasoning-focused SAE corpus from .npz traces")
    ap.add_argument("--npz_dir", required=True, help="Directory containing .npz traces")
    ap.add_argument("--out_dir", required=True, help="Output root, e.g., run/rsae_corpus")
    ap.add_argument("--dilate", type=int, default=2, help="Reasoning mask dilation window")
    ap.add_argument("--neg_ratio", type=float, default=0.1, help="Fraction of non-reasoning negatives to include")
    ap.add_argument("--max_per_file", type=int, default=64, help="Max tokens per file per layer")
    ap.add_argument("--shard_size", type=int, default=20000, help="Rows per shard per layer")
    args = ap.parse_args()

    npz_root = Path(args.npz_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 先探测一份文件，拿到 L、D
    probe = next(iter(iter_npz_files(npz_root)), None)
    if probe is None:
        print("No npz found"); return
    with np.load(probe, allow_pickle=False) as data:
        T, L, D = data["acts"].shape
    print(f"Detected shape template: T=?, L={L}, D={D}")

    # 为每个层准备缓冲与统计
    buffers = {l: [] for l in range(L)}        # 每层累积样本（多文件拼起来），后面切 shard
    count_per_file = {l: 0 for l in range(L)}  # 仅用于打印调试

    # 遍历所有 npz 文件
    for idx, fp in enumerate(iter_npz_files(npz_root), start=1):
        with np.load(fp, allow_pickle=False) as data:
            acts = data["acts"]            # [T, L, D]  float16
            mask = data["reasoning_mask"]  # [T]
            if acts.shape[1] != L or acts.shape[2] != D:
                print(f"[WARN] shape mismatch in {fp}, skip"); continue

            # 1) 逐题逐层中心化（去除题材基线）
            acts_c = per_layer_centering(acts)

            # 2) 选择样本索引
            m = mask.astype(bool)
            m = dilate_mask(m, args.dilate)

            pos_idx = np.where(m)[0]
            neg_idx = np.where(~m)[0]

            # 限制每题每层的最多采样数（保证均衡）
            # 先决定本题的目标采样数上限
            per_file_cap_pos = int(args.max_per_file * (1.0 - args.neg_ratio))
            per_file_cap_neg = args.max_per_file - per_file_cap_pos

            if len(pos_idx) > per_file_cap_pos:
                pos_idx = np.random.choice(pos_idx, size=per_file_cap_pos, replace=False)
            if len(neg_idx) > 0 and per_file_cap_neg > 0:
                kneg = min(per_file_cap_neg, int(len(pos_idx) * args.neg_ratio / (1 - args.neg_ratio) + 1))
                kneg = min(kneg, len(neg_idx))
                neg_idx = np.random.choice(neg_idx, size=kneg, replace=False)
            else:
                neg_idx = np.array([], dtype=int)

            sel_idx = np.sort(np.concatenate([pos_idx, neg_idx], axis=0))  # [S]

            # 3) 将本题选中的 token 行，按层拆开并累积
            # acts_c[sel_idx, l, :] -> 形状 [S, D]
            for l in range(L):
                X = acts_c[sel_idx, l, :].astype(np.float32)  # 用 float32 做统计更稳
                buffers[l].append(X)
                count_per_file[l] += X.shape[0]

        if idx % 50 == 0:
            print(f"Processed {idx} files ...")

    # 4) 对每层合并与标准化、切 shard、写盘
    for l in range(L):
        if not buffers[l]:
            print(f"[WARN] layer {l} has no data, skip"); continue
        X = np.concatenate(buffers[l], axis=0)  # [N, D]
        print(f"Layer {l}: collected {X.shape[0]} rows")

        # 统计 RMS 并缩放（可选 z-score，这里给 RMS，实操中很稳）
        rms = compute_rms(X, axis=0)  # [D]
        X_scaled = X / rms

        # 输出目录
        layer_dir = out_root / f"layer{l}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        # 切 shard
        N = X_scaled.shape[0]
        shard = 0
        for i in range(0, N, args.shard_size):
            j = min(N, i + args.shard_size)
            np.savez_compressed(layer_dir / f"shard-{shard:04d}.npz", X=X_scaled[i:j].astype(np.float16))
            shard += 1

        # 保存统计信息（推理/评估阶段需要同样的缩放）
        stats = {
            "layer_index": l,
            "D": int(D),
            "rows": int(N),
            "scaler": "rms",
            "rms": rms.astype(np.float32).tolist(),
            "prep": {
                "per_file_centering": True,
                "dilate": args.dilate,
                "neg_ratio": args.neg_ratio,
                "max_per_file": args.max_per_file,
            },
        }
        (layer_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Layer {l}: wrote {shard} shards to {layer_dir}")

if __name__ == "__main__":
    main()
