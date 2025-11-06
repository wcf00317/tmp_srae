# src/crb/instrumentation/activation_recorder.py
from __future__ import annotations
from typing import List, Dict, Any
import torch
import torch.nn as nn

class ActivationRecorder:
    """
    在指定的 transformer blocks 上注册 forward hook。
    每次 forward（通常对应一步解码）都会捕获这些 block 的输出 hidden_states。
    - capture="input" 时记录模块输入（接近残差流）
    - capture="output" 时记录模块输出
    """
    def __init__(self, blocks: List[nn.Module], capture: str = "output", dtype: torch.dtype = torch.bfloat16):
        assert capture in ("input", "output")
        self.blocks = blocks
        self.capture = capture
        self.dtype = dtype
        self.hooks = []
        self._current_step: List[torch.Tensor] = []

        def _mk_hook(idx: int):
            def _hook(module, inputs, output):
                if self.capture == "input":
                    x = inputs[0]
                else:
                    x = output
                # 只保留最后一个 token 的表征（增量解码时 batch=1、seq=1）
                # 若 seq>1，取最后位置
                if x.dim() == 3:
                    x = x[:, -1, :]  # (B, D)
                elif x.dim() == 2:
                    pass  # (B, D)
                else:
                    raise RuntimeError(f"不支持的张量形状: {x.shape}")
                self._current_step.append(x.detach().to("cpu", dtype=self.dtype))
            return _hook

        for i, blk in enumerate(self.blocks):
            self.hooks.append(blk.register_forward_hook(_mk_hook(i)))

    def start_step(self):
        self._current_step = []

    def pop_step_features(self) -> torch.Tensor:
        """
        返回形状 [L, D] 的张量（已在 CPU 上，dtype=初始化传入的 dtype）
        """
        if not self._current_step:
            raise RuntimeError("本步尚未捕获到任何激活。请确认 hook 是否安装成功以及 forward 是否发生。")
        # [B, D] 按 block 顺序栈在一起，且 batch=1 -> squeeze(0)
        feats = [t.squeeze(0) for t in self._current_step]
        return torch.stack(feats, dim=0)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
