"""回答内容与伪安全结构关联分析的轻量工具。

这里只从回答正文提取特征，明确不读取 ``toxic`` 或 ``toxicA``。这些字段参与
ASR 标签生成，若再作为输入会造成直接标签泄漏。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


REFUSAL_PATTERNS: tuple[str, ...] = (
    r"^\s*(?:i(?:'m| am) sorry|sorry)[,，.:;!\s-]*",
    r"^\s*i (?:cannot|can't|won't|will not|am unable to)[^.!?\n]{0,180}[.!?]?\s*",
    r"^\s*as an ai(?: language model)?[^.!?\n]{0,220}[.!?]?\s*",
    r"^\s*i (?:must|have to) (?:decline|refuse)[^.!?\n]{0,180}[.!?]?\s*",
)

REFUSAL_TERMS: tuple[str, ...] = (
    "i'm sorry",
    "i am sorry",
    "i cannot",
    "i can't",
    "i won't",
    "unable to",
    "cannot assist",
    "can't assist",
    "cannot help",
    "can't help",
    "must refuse",
    "illegal",
    "harmful",
    "dangerous",
    "instead",
)

FEATURE_NAMES: tuple[str, ...] = (
    "回答字符数",
    "回答词数",
    "回答行数",
    "回答句数",
    "实质内容字符数",
    "实质内容占比",
    "拒绝词命中数",
    "是否拒绝开头",
    "代码块数",
    "列表项数",
    "问号数",
    "感叹号数",
    "词汇多样性",
    "平均词长",
)


@dataclass(frozen=True)
class AnswerView:
    full_text: str
    substantive_text: str
    features: np.ndarray


def strip_refusal_prefix(text: str) -> tuple[str, bool]:
    """删除开头常见拒绝套话，返回剩余文本及是否命中过模式。"""

    cleaned = text.strip()
    matched = False
    for pattern in REFUSAL_PATTERNS:
        updated, count = re.subn(pattern, "", cleaned, count=1, flags=re.IGNORECASE)
        if count:
            cleaned = updated.strip()
            matched = True
    return cleaned, matched


def answer_view(text: str) -> AnswerView:
    """提取单条回答的可解释特征。"""

    full = str(text or "").strip()
    substantive, refusal_prefix = strip_refusal_prefix(full)
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[\u4e00-\u9fff]", full)
    lowered = full.lower()
    unique_ratio = len({word.lower() for word in words}) / max(1, len(words))
    mean_word_len = sum(len(word) for word in words) / max(1, len(words))
    refusal_hits = sum(lowered.count(term) for term in REFUSAL_TERMS)
    features = np.asarray(
        [
            len(full),
            len(words),
            max(1, len(full.splitlines())) if full else 0,
            len(re.findall(r"[.!?。！？]+", full)),
            len(substantive),
            len(substantive) / max(1, len(full)),
            refusal_hits,
            int(refusal_prefix),
            full.count("```"),
            len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", full)),
            full.count("?") + full.count("？"),
            full.count("!") + full.count("！"),
            unique_ratio,
            mean_word_len,
        ],
        dtype=np.float32,
    )
    return AnswerView(full_text=full, substantive_text=substantive, features=features)


def aggregate_answers(
    answers: Sequence[dict],
    sample_size: int,
    rng: np.random.Generator,
    long_threshold: int = 500,
    max_text_chars: int = 12000,
) -> tuple[np.ndarray, str, dict[str, float]]:
    """抽取少量回答并聚合特征，同时构造去拒绝前缀的文本表示。"""

    texts = [str(item.get("A", "")) for item in answers if isinstance(item, dict)]
    if not texts:
        views = [answer_view("")]
    else:
        count = min(sample_size, len(texts))
        indices = rng.choice(len(texts), size=count, replace=False)
        views = [answer_view(texts[int(index)]) for index in indices]
    matrix = np.stack([view.features for view in views])
    summary = np.concatenate([matrix.mean(axis=0), matrix.std(axis=0), matrix.min(axis=0), matrix.max(axis=0)])
    substantive = "\n\n<回答分隔>\n\n".join(view.substantive_text for view in views)[:max_text_chars]
    extras = {
        "sampled_answers": float(len(views)),
        "sample_long_ratio": float(np.mean([len(view.full_text) >= long_threshold for view in views])),
        "sample_refusal_ratio": float(np.mean([view.features[7] for view in views])),
    }
    return summary.astype(np.float32), substantive, extras


def pairwise_accuracy_by_group(
    targets: np.ndarray,
    predictions: np.ndarray,
    groups: Iterable[tuple[str, int]],
    min_target_delta: float = 0.0,
) -> float:
    """计算同攻击方案、同问题内的序关系准确率。"""

    grouped: dict[tuple[str, int], list[int]] = {}
    for index, group in enumerate(groups):
        grouped.setdefault(group, []).append(index)
    correct = 0
    total = 0
    for indices in grouped.values():
        for pos, left in enumerate(indices):
            for right in indices[pos + 1 :]:
                delta = float(targets[left] - targets[right])
                if abs(delta) <= min_target_delta:
                    continue
                pred_delta = float(predictions[left] - predictions[right])
                if pred_delta == 0:
                    correct += 0.5
                elif (pred_delta > 0) == (delta > 0):
                    correct += 1
                total += 1
    return float(correct / total) if total else float("nan")


def pairwise_accuracy_by_group_macro(
    targets: np.ndarray,
    predictions: np.ndarray,
    groups: Iterable[tuple[str, int]],
    min_target_delta: float = 0.0,
) -> float:
    """先在每个攻击方案/问题内算序准确率，再对问题做等权平均。"""

    grouped: dict[tuple[str, int], list[int]] = {}
    for index, group in enumerate(groups):
        grouped.setdefault(group, []).append(index)
    accuracies = []
    for group, indices in grouped.items():
        local_targets = targets[indices]
        local_predictions = predictions[indices]
        local_groups = [group] * len(indices)
        accuracy = pairwise_accuracy_by_group(local_targets, local_predictions, local_groups, min_target_delta)
        if np.isfinite(accuracy):
            accuracies.append(accuracy)
    return float(np.mean(accuracies)) if accuracies else float("nan")
