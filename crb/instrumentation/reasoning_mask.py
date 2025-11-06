# src/crb/instrumentation/reasoning_mask.py
from __future__ import annotations
import re
from typing import List
from transformers import PreTrainedTokenizer

_OP_RE = re.compile(r"[=\+\-\*/\^\(\)\[\]\{\}<>]|>=|<=|==|!=")
_KEYWORDS = [
    "因此", "所以", "令", "设", "若", "则", "因", "故", "检验", "检查",
    "更新", "带入", "代入", "计算", "结论", "推得", "可得", "于是", "求",
    "because", "therefore", "let", "assume", "thus", "check", "update", "substitute"
]
_NUM_RE = re.compile(r"\d")

def compute_reasoning_mask(gen_ids: List[int], tokenizer: PreTrainedTokenizer, window: int = 0) -> List[bool]:
    """
    对生成的每个 token 设定一个布尔值，表示是否属于“推理窗口”。
    规则：token 文本中包含运算符/数字/关键词即视为 True。
    window>0 时对 True 位置做邻域膨胀。
    """
    toks = [tokenizer.decode([tid], skip_special_tokens=True) for tid in gen_ids]
    mask = []
    for t in toks:
        hit = bool(_OP_RE.search(t)) or bool(_NUM_RE.search(t)) or any(k in t for k in _KEYWORDS)
        mask.append(hit)
    if window > 0:
        n = len(mask)
        dilated = [False] * n
        for i, v in enumerate(mask):
            if v:
                lo = max(0, i - window)
                hi = min(n, i + 1 + window)
                for j in range(lo, hi):
                    dilated[j] = True
        return dilated
    return mask
