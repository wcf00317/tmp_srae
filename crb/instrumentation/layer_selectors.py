# src/crb/instrumentation/layer_selectors.py
from __future__ import annotations
import re
from typing import List, Tuple, Dict
import torch.nn as nn

def get_transformer_blocks(model: nn.Module) -> Tuple[List[nn.Module], List[str]]:
    """
    兼容 Qwen3 模型常见的层路径：
    - model.model.layers[i]
    - transformer.h[i]  (兼容一些 LLaMA 系模型)
    返回 (blocks, names)
    """
    candidates = []
    names = []

    # 1) 常见的 Qwen3 路径
    try:
        layers = getattr(getattr(model, "model"), "layers")
        if isinstance(layers, (list, tuple)) and len(layers) > 0:
            for i, block in enumerate(layers):
                candidates.append(block)
                names.append(f"model.model.layers.{i}")
            return candidates, names
    except Exception:
        pass

    # 2) 兼容 LLama 风格
    try:
        h = getattr(getattr(model, "transformer"), "h")
        if isinstance(h, (list, tuple)) and len(h) > 0:
            for i, block in enumerate(h):
                candidates.append(block)
                names.append(f"transformer.h.{i}")
            return candidates, names
    except Exception:
        pass

    # 3) 兜底：正则匹配以 .layers.N 或 .h.N 结尾的模块名
    for name, module in model.named_modules():
        if re.search(r"(layers|h)\.\d+$", name):
            candidates.append(module)
            names.append(name)

    if not candidates:
        raise RuntimeError("未能自动识别 transformer blocks。请检查模型结构或手动指定。")
    return candidates, names

def pick_blocks_by_indices(model: nn.Module, indices: List[int]):
    blocks, names = get_transformer_blocks(model)
    picked_modules, picked_names = [], []
    for i in indices:
        if i < 0 or i >= len(blocks):
            raise IndexError(f"层索引超界: {i}（共有 {len(blocks)} 层）")
        picked_modules.append(blocks[i])
        picked_names.append(names[i])
    return picked_modules, picked_names
