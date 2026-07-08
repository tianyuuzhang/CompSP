"""监督伪安全方向学习与投影工具。

本模块服务于“安全结构可学习性”实验：先用白盒 Llama 最后一层 hidden state
和 ALR/ASR 监督学习一维线性方向，再把每条指令投影到该方向上，构造新的
pairwise 训练目标。这里的方向是实验定义的伪安全方向，不等同于真实安全向量。
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from compsp.asr_dataset import ASRRecord, iter_records

TEST_QIDS: tuple[int, ...] = (9, 30, 73, 10, 37, 15, 42, 1, 50, 41, 45, 65, 70, 72, 21, 55, 56, 78, 48, 76)
TRAIN_QIDS: tuple[int, ...] = tuple(i for i in range(80) if i not in set(TEST_QIDS))
DEFAULT_DATASET_KEYS: tuple[str, ...] = ("jbb-llama-ofa", "jbb-llama-pair", "jbb-llama-drattack")


@dataclass(frozen=True)
class DirectionRecord:
    dataset_key: str
    split: str
    question_id: int
    item_index: int
    round: int
    q1: str
    asr: float
    alr: float
    tx: int
    txl: int
    tot_qa: int


@dataclass(frozen=True)
class DirectionFitReport:
    metric: str
    selected_alpha: float
    train_rows: int
    test_rows: int
    train_questions: list[int]
    test_questions: list[int]
    dataset_keys: list[str]
    cv_results: list[dict[str, float]]


def parse_csv_ints(text: str | None, default: Sequence[int]) -> list[int]:
    if text is None or not text.strip():
        return list(default)
    return [int(x) for x in text.split(",") if x.strip()]


def parse_dataset_keys(text: str | None, default: Sequence[str] = DEFAULT_DATASET_KEYS) -> list[str]:
    if text is None or not text.strip():
        return list(default)
    return [x.strip() for x in text.split(",") if x.strip()]


def collect_direction_records(
    dataset_keys: Sequence[str],
    train_qids: Sequence[int] = TRAIN_QIDS,
    test_qids: Sequence[int] = TEST_QIDS,
    max_items_per_question: int | None = None,
) -> list[DirectionRecord]:
    """按固定 60/20 题划分收集 q1 与 ASR/ALR 元数据。"""

    rows: list[DirectionRecord] = []
    split_by_qid = {qid: "train" for qid in train_qids} | {qid: "test" for qid in test_qids}
    for dataset_key in dataset_keys:
        for split, qids in (("train", train_qids), ("test", test_qids)):
            for qid in qids:
                for idx, record in enumerate(iter_records(dataset_key, [qid])):
                    if not record.q1:
                        continue
                    if max_items_per_question is not None and idx >= max_items_per_question:
                        break
                    rows.append(
                        DirectionRecord(
                            dataset_key=dataset_key,
                            split=split_by_qid.get(record.question_id, split),
                            question_id=record.question_id,
                            item_index=idx,
                            round=record.round,
                            q1=record.q1,
                            asr=record.asr,
                            alr=record.alr,
                            tx=record.tx,
                            txl=record.txl,
                            tot_qa=record.tot_qa,
                        )
                    )
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    raise ValueError(f"未知相关方法: {method}")


def fit_ridge_direction(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    alphas: Sequence[float],
    n_splits: int = 5,
) -> tuple[Ridge, DirectionFitReport, dict[float, list[float]]]:
    """使用按问题分组的交叉验证选择 Ridge alpha。"""

    unique_groups = np.unique(groups)
    actual_splits = min(n_splits, len(unique_groups))
    if actual_splits < 2:
        raise ValueError("训练问题数不足，无法做 GroupKFold 选择 alpha")

    cv = GroupKFold(n_splits=actual_splits)
    scores_by_alpha: dict[float, list[float]] = {float(alpha): [] for alpha in alphas}
    for alpha in alphas:
        alpha = float(alpha)
        for train_idx, val_idx in cv.split(x_train, y_train, groups):
            model = Ridge(alpha=alpha, fit_intercept=True, random_state=42)
            model.fit(x_train[train_idx], y_train[train_idx])
            pred = model.predict(x_train[val_idx])
            scores_by_alpha[alpha].append(safe_corr(pred, y_train[val_idx], "spearman"))

    def mean_score(alpha: float) -> float:
        values = np.array(scores_by_alpha[alpha], dtype=float)
        if np.all(np.isnan(values)):
            return float( -inf)
        return float(np.nanmean(values))

    selected_alpha = max(scores_by_alpha, key=mean_score)
    final_model = Ridge(alpha=selected_alpha, fit_intercept=True, random_state=42)
    final_model.fit(x_train, y_train)
    report = DirectionFitReport(
        metric="",
        selected_alpha=float(selected_alpha),
        train_rows=int(len(y_train)),
        test_rows=0,
        train_questions=sorted(map(int, unique_groups.tolist())),
        test_questions=[],
        dataset_keys=[],
        cv_results=[
            {
                "alpha": float(alpha),
                "mean_spearman": mean_score(float(alpha)),
                "folds": float(len(scores_by_alpha[float(alpha)])),
            }
            for alpha in alphas
        ],
    )
    return final_model, report, scores_by_alpha


def project_with_ridge(model: Ridge, x: np.ndarray) -> np.ndarray:
    """返回 Ridge 方向的一维投影；intercept 不影响序关系，但保留便于复现实验。"""

    return model.predict(x)


def save_direction(path: str | Path, model: Ridge, report: DirectionFitReport, extra: dict | None = None) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "coef": torch.tensor(model.coef_, dtype=torch.float32),
        "intercept": float(model.intercept_),
        "report": asdict(report),
        "extra": extra or {},
        "method": "last_layer_last_token_ridge",
    }
    torch.save(payload, out)


def load_direction(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu")


def normalize_scores_by_question(rows: list[dict], score_key: str = "pseudo_score") -> list[dict]:
    """在每个 dataset/question 内把投影归一到 [0, 1]，便于统一 min_delta。"""

    by_group: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        by_group.setdefault((row["dataset_key"], int(row["question_id"])), []).append(row)
    out: list[dict] = []
    for _, items in by_group.items():
        values = np.array([float(item[score_key]) for item in items], dtype=float)
        lo, hi = float(values.min()), float(values.max())
        denom = hi - lo
        for item in items:
            new_item = dict(item)
            new_item["pseudo_score_norm"] = 0.5 if denom == 0 else (float(item[score_key]) - lo) / denom
            out.append(new_item)
    return out


def build_pseudo_pairs(
    rows: list[dict],
    split: str,
    min_delta: float = 0.15,
    max_pairs_per_question: int | None = 2000,
    seed: int = 42,
) -> list[dict]:
    """由伪安全投影构造同题内 pairwise 样本。"""

    rng = random.Random(seed)
    usable = normalize_scores_by_question([row for row in rows if row.get("split") == split])
    by_group: dict[tuple[str, int], list[dict]] = {}
    for row in usable:
        by_group.setdefault((row["dataset_key"], int(row["question_id"])), []).append(row)

    pairs: list[dict] = []
    for (dataset_key, question_id), items in sorted(by_group.items()):
        group_pairs = []
        for left, right in combinations(items, 2):
            left_score = float(left["pseudo_score_norm"])
            right_score = float(right["pseudo_score_norm"])
            delta = left_score - right_score
            if abs(delta) <= min_delta:
                continue
            label = 1 if delta > 0 else 0
            a, b = left, right
            if rng.random() < 0.5:
                a, b = right, left
                label = 1 - label
            group_pairs.append(
                {
                    "text": f'Text A: "{a.get("q1", "")}"\n---\nText B: "{b.get("q1", "")}"\n',
                    "label": int(label),
                    "dataset_key": dataset_key,
                    "questionID": question_id,
                    "split": split,
                    "a_item_index": int(a["item_index"]),
                    "b_item_index": int(b["item_index"]),
                    "a_pseudo_score": float(a["pseudo_score"]),
                    "b_pseudo_score": float(b["pseudo_score"]),
                    "a_pseudo_score_norm": float(a["pseudo_score_norm"]),
                    "b_pseudo_score_norm": float(b["pseudo_score_norm"]),
                    "delta_norm_abs": float(abs(delta)),
                    "a_alr": float(a["alr"]),
                    "b_alr": float(b["alr"]),
                    "a_asr": float(a["asr"]),
                    "b_asr": float(b["asr"]),
                }
            )
        if max_pairs_per_question is not None and len(group_pairs) > max_pairs_per_question:
            group_pairs = rng.sample(group_pairs, max_pairs_per_question)
        pairs.extend(group_pairs)
    rng.shuffle(pairs)
    return pairs
