# src/crb/rsae/train_rsae.py
from __future__ import annotations
import os, json, math, argparse
from pathlib import Path
from typing import Optional, Dict
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from crb.rsae.model import SparseAutoencoder
from crb.rsae.dataset import make_dataloaders

def try_load_pretrained(model: nn.Module, ckpt_path: Optional[str]) -> None:
    if not ckpt_path: return
    p = Path(ckpt_path)
    if not p.exists():
        print(f"[warmstart] file not found: {p} (skip)"); return
    try:
        sd = torch.load(p, map_location="cpu")
        if "state_dict" in sd: sd = sd["state_dict"]
        msd = model.state_dict()
        # 只加载形状匹配的键
        keep = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
        missing = [k for k in msd.keys() if k not in keep]
        print(f"[warmstart] load {len(keep)}/{len(msd)} tensors from {p.name}")
        model.load_state_dict({**msd, **keep}, strict=False)
    except Exception as e:
        print(f"[warmstart] failed: {e} (skip)")

def train_one(cfg: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_dir = cfg["data"]["layer_dir"]
    save_dir = Path(cfg["io"]["save_dir"]); save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_dataloaders(
        layer_dir, batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"].get("num_workers", 0),
        val_ratio=cfg["train"].get("val_ratio", 0.02)
    )

    # 从 stats.json 读入 D
    stats_path = Path(layer_dir) / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    d_in = stats["D"]
    d_code = int(cfg["model"]["d_code"])
    tied = bool(cfg["model"].get("tied", False))

    model = SparseAutoencoder(d_in=d_in, d_code=d_code, tied=tied).to(device)
    try_load_pretrained(model, cfg["model"].get("warmstart_path"))

    opt = AdamW(model.parameters(), lr=cfg["train"]["lr"], betas=(0.9, 0.95), weight_decay=cfg["train"].get("wd", 0.0))
    scaler = GradScaler(enabled=cfg["train"].get("amp", True))
    mse = nn.MSELoss(reduction="mean")

    # L1 热身设置
    l1_max = float(cfg["loss"]["l1_lambda"])
    warm_steps = int(cfg["loss"].get("l1_warmup_steps", 0))

    best_val = math.inf
    log = {"steps": []}
    step = 0
    epochs = int(cfg["train"]["epochs"])

    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch.to(device, dtype=torch.float32)  # X 已经标准化
            with autocast(enabled=cfg["train"].get("amp", True)):
                out = model(x)
                recon = mse(out["x_hat"], x)
                # 线性抬升 l1 系数
                l1_now = l1_max if warm_steps <= 0 else min(l1_max, l1_max * (step / warm_steps))
                l1 = out["z"].abs().mean()
                loss = recon + l1_now * l1

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                l0 = model.l0_per_example(out["z"]).mean().item()
            if step % cfg["io"].get("log_every", 50) == 0:
                print(f"[ep {ep}] step {step}  recon={recon.item():.6f}  l1={l1.item():.6f}  "
                      f"λ={l1_now:.3g}  L0~{l0:.1f}")
                log["steps"].append({"step": step, "recon": float(recon.item()), "l1": float(l1.item()),
                                     "lambda": float(l1_now), "L0": float(l0)})

            if val_loader and step % cfg["io"].get("val_every", 500) == 0 and step > 0:
                model.eval()
                v_loss, v_cnt = 0.0, 0
                with torch.no_grad(), autocast(enabled=cfg["train"].get("amp", True)):
                    for vbatch in val_loader:
                        vx = vbatch.to(device, dtype=torch.float32)
                        vout = model(vx)
                        v_loss += mse(vout["x_hat"], vx).item() * vx.size(0)
                        v_cnt += vx.size(0)
                v_mse = v_loss / max(1, v_cnt)
                print(f"  [val] mse={v_mse:.6f}  (best={best_val:.6f})")
                if v_mse < best_val:
                    best_val = v_mse
                    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "best_val_mse": best_val},
                               save_dir / "best.pt")
                model.train()

            step += 1
            if step >= cfg["train"].get("max_steps", 0) > 0:
                break
        if step >= cfg["train"].get("max_steps", 0) > 0:
            break

    # 保存最终
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "best_val_mse": best_val},
               save_dir / "last.pt")
    (save_dir / "train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] saved to {save_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer_dir", required=True, help="e.g., run/rsae_corpus/layer1")
    ap.add_argument("--save_dir", required=True, help="e.g., run/rsae_ckpts/layer1_v0")
    ap.add_argument("--d_code", type=int, default=None, help="code size; default=4x D")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--l1", type=float, default=1e-3)
    ap.add_argument("--l1_warmup_steps", type=int, default=2000)
    ap.add_argument("--tied", action="store_true")
    ap.add_argument("--warmstart", type=str, default=None, help="optional .pt path")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    # 读取 D 以便默认 d_code=4*D
    stats = json.loads((Path(args.layer_dir) / "stats.json").read_text(encoding="utf-8"))
    d_in = stats["D"]
    d_code = args.d_code if args.d_code is not None else int(4 * d_in)

    cfg = {
        "data": {"layer_dir": args.layer_dir},
        "model": {"d_code": d_code, "tied": bool(args.tied), "warmstart_path": args.warmstart},
        "train": {
            "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
            "max_steps": args.max_steps, "amp": bool(args.amp), "val_every": args.val_every
        },
        "loss": {"l1_lambda": args.l1, "l1_warmup_steps": args.l1_warmup_steps},
        "io": {"save_dir": args.save_dir, "log_every": args.log_every}
    }
    (Path(args.save_dir) / "cfg.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    train_one(cfg)

if __name__ == "__main__":
    main()
