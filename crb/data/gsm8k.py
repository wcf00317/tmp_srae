# src/crb/data/gsm8k.py
from __future__ import annotations
import re
from typing import Iterator, Dict, Any, Optional
from datasets import load_dataset

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

def extract_final_number(text: str) -> Optional[str]:
    """
    GSM8K 标准答案通常包含 '#### 123'；也支持兜底在文本尾部抓取最后一个数字。
    返回字符串形式（去前导零），None 表示没找到。
    """
    if "####" in text:
        tail = text.split("####")[-1]
        m = _NUM_RE.search(tail)
        if m:
            return str(float(m.group())).rstrip("0").rstrip(".")
    # 兜底：取文本中最后一个数字
    ms = list(_NUM_RE.finditer(text))
    if ms:
        return str(float(ms[-1].group())).rstrip("0").rstrip(".")
    return None

def normalize_gold(answer: str) -> Optional[str]:
    return extract_final_number(answer)

def equals_num(pred: Optional[str], gold: Optional[str]) -> bool:
    if pred is None or gold is None:
        return False
    return pred == gold

def iter_gsm8k(split: str = "test", limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """
    产出字典：{id, question, answer, gold_norm}
    """
    ds = load_dataset("gsm8k", "main")[split]
    count = 0
    for ex in ds:
        qid = ex.get("id") or ex.get("question", "")[:64]
        gold = normalize_gold(ex["answer"])
        yield {
            "id": qid,
            "question": ex["question"],
            "answer": ex["answer"],
            "gold_norm": gold,
        }
        count += 1
        if limit is not None and count >= limit:
            break
