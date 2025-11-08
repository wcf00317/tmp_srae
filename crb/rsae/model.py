# src/crb/rsae/model.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Optional, Dict

class SparseAutoencoder(nn.Module):
    """
    x (B, D) -> z = ReLU(x W_e + b_e) (B, M) -> x_hat = z W_d + b_d  (B, D)
    训练目标: MSE(x_hat, x) + lambda * mean(|z|)
    """
    def __init__(self, d_in: int, d_code: int, tied: bool = False):
        super().__init__()
        self.d_in = d_in
        self.d_code = d_code
        self.tied = tied

        self.encoder = nn.Linear(d_in, d_code, bias=True)
        if tied:
            self.decoder_bias = nn.Parameter(torch.zeros(d_in))
            # decoder 权重由 encoder.weight^T 给出
        else:
            self.decoder = nn.Linear(d_code, d_in, bias=True)

        self.act = nn.ReLU(inplace=False)
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming for encoder; small init for decoder
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        if self.encoder.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.encoder.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.encoder.bias, -bound, +bound)
        if self.tied:
            nn.init.zeros_(self.decoder_bias)
        else:
            nn.init.xavier_uniform_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.tied:
            # x_hat = z @ W_e^T + b_d
            return torch.addmm(self.decoder_bias, z, self.encoder.weight.t())
        else:
            return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_pre = self.encoder(x)
        z = self.act(z_pre)
        x_hat = self.decode(z)
        return {"x_hat": x_hat, "z": z, "z_pre": z_pre}

    @torch.no_grad()
    def l0_per_example(self, z: torch.Tensor, thr: float = 1e-8) -> torch.Tensor:
        # 每个样本非零（>thr）的单元数
        return (z > thr).float().sum(dim=1)
