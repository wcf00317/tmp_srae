# src/crb/rsae/dataset.py
from __future__ import annotations
import json, random
from pathlib import Path
from typing import Iterator, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

def list_shards(layer_dir: str) -> List[Path]:
    p = Path(layer_dir)
    return sorted(p.glob("shard-*.npz"))

class NpzShardDataset(IterableDataset):
    """
    逐 shard 流式读取，单次只保留一个 shard 在内存。
    每个 shard: { X: [N, D] float16 } —— 已由准备脚本做过标准化
    """
    def __init__(self, layer_dir: str, shuffle: bool = True, repeat: bool = False):
        super().__init__()
        self.layer_dir = Path(layer_dir)
        self.shards = list_shards(layer_dir)
        if not self.shards:
            raise FileNotFoundError(f"No shards found in {layer_dir}")
        self.shuffle = shuffle
        self.repeat = repeat

    def __iter__(self) -> Iterator[torch.Tensor]:
        order = list(range(len(self.shards)))
        rng = random.Random()
        while True:
            if self.shuffle:
                rng.shuffle(order)
            for idx in order:
                with np.load(self.shards[idx], allow_pickle=False) as data:
                    X = data["X"]  # [N, D], float16
                    # 打乱当前 shard 行顺序
                    perm = np.arange(X.shape[0])
                    if self.shuffle:
                        rng.shuffle(perm.tolist())
                    for i in perm:
                        yield torch.from_numpy(X[i].astype(np.float32, copy=False))
            if not self.repeat:
                break

def make_dataloaders(layer_dir: str, batch_size: int, num_workers: int = 0,
                     val_ratio: float = 0.02) -> Tuple[DataLoader, Optional[DataLoader]]:
    shards = list_shards(layer_dir)
    n = len(shards)
    n_val = max(1, int(n * val_ratio)) if n >= 10 else 0
    if n_val > 0:
        train_shards = shards[:-n_val]
        val_shards = shards[-n_val:]
    else:
        train_shards, val_shards = shards, []

    def ds_from(shs, repeat):
        tmp = Path(layer_dir) / ".tmp_subset.txt"
        tmp.write_text("\n".join(str(s) for s in shs), encoding="utf-8")
        class SubsetDataset(NpzShardDataset):
            def __init__(self, fn_list: Path, shuffle=True, repeat=False):
                self.layer_dir = Path(layer_dir)
                self.shards = [Path(x) for x in fn_list.read_text(encoding="utf-8").splitlines() if x.strip()]
                self.shuffle = shuffle; self.repeat = repeat
                if not self.shards: raise FileNotFoundError("Empty subset")
        return SubsetDataset(tmp, shuffle=True, repeat=repeat)

    train_ds = ds_from(train_shards, repeat=True)
    val_ds = ds_from(val_shards, repeat=False) if n_val > 0 else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers) if val_ds else None
    return train_loader, val_loader
