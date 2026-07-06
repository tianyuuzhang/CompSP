"""ASR/ALR txfilled 数据集的标准加载器和校验器。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from .asr_registry import ASRDatasetSpec, get_asr_spec

CORE_FIELDS = {"questionID", "question", "round", "totQA", "Ans", "q1", "txl", "tx"}
OFA_FIELDS = CORE_FIELDS | {"q0", "a0"}
ANSWER_FIELDS = {"A", "len", "toxic", "toxicA"}


@dataclass(frozen=True)
class ASRRecord:
    dataset_key: str
    question_id: int
    question: str
    round: int
    q1: str
    tx: int
    txl: int
    tot_qa: int
    answers: list[dict[str, Any]]
    q0: str | None = None
    a0: str | None = None
    target: str | None = None
    raw: dict[str, Any] | None = None

    @property
    def asr(self) -> float:
        return self.tx / self.tot_qa if self.tot_qa else 0.0

    @property
    def alr(self) -> float:
        return self.txl / self.tot_qa if self.tot_qa else 0.0


def parse_question_id(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("Q_"):
            text = text[2:]
        return int(text)
    raise TypeError(f"不支持的 questionID 类型: {type(value).__name__}")


def discover_question_ids(spec: ASRDatasetSpec) -> list[int]:
    ids: list[int] = []
    if not spec.root_dir.exists():
        return ids
    for child in spec.root_dir.glob("Q_*"):
        if not child.is_dir():
            continue
        try:
            ids.append(int(child.name.split("_", 1)[1]))
        except ValueError:
            continue
    return sorted(set(ids))


def load_txfilled_file(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"期望 JSON list: {path}")
    return data


def normalize_record(dataset_key: str, item: dict[str, Any]) -> ASRRecord:
    return ASRRecord(
        dataset_key=dataset_key,
        question_id=parse_question_id(item["questionID"]),
        question=str(item["question"]),
        round=int(item["round"]),
        q1=str(item["q1"]),
        tx=int(item["tx"]),
        txl=int(item["txl"]),
        tot_qa=int(item["totQA"]),
        answers=list(item.get("Ans", [])),
        q0=item.get("q0"),
        a0=item.get("a0"),
        target=item.get("target"),
        raw=item,
    )


def iter_records(dataset_key: str, question_ids: Iterable[int] | None = None) -> Iterable[ASRRecord]:
    spec = get_asr_spec(dataset_key)
    ids = list(question_ids) if question_ids is not None else discover_question_ids(spec)
    for question_id in ids:
        path = spec.path_for(question_id)
        if not path.exists():
            continue
        for item in load_txfilled_file(path):
            yield normalize_record(dataset_key, item)


def make_pairwise_examples(
    records: Iterable[ASRRecord],
    metric: str = "asr",
    min_delta: float = 0.15,
) -> list[dict[str, Any]]:
    """为 CompSP 训练/评估构造同问题内的 pairwise 比较样本。

    `metric` 可以是 `asr` 或 `alr`。绝对指标差 <= `min_delta` 的样本会被移除，
    对应论文中的噪声控制规则。
    """

    if metric not in {"asr", "alr"}:
        raise ValueError("metric 必须是 'asr' 或 'alr'")
    by_question: dict[int, list[ASRRecord]] = {}
    for record in records:
        by_question.setdefault(record.question_id, []).append(record)

    examples: list[dict[str, Any]] = []
    for question_id, items in by_question.items():
        for left, right in combinations(items, 2):
            left_score = getattr(left, metric)
            right_score = getattr(right, metric)
            delta = left_score - right_score
            if abs(delta) <= min_delta:
                continue
            examples.append(
                {
                    "questionID": question_id,
                    "left_round": left.round,
                    "right_round": right.round,
                    "left_q1": left.q1,
                    "right_q1": right.q1,
                    "metric": metric,
                    "left_score": left_score,
                    "right_score": right_score,
                    "label": 1 if delta > 0 else 0,
                    "dataset_key": left.dataset_key,
                }
            )
    return examples


def validate_item(item: Any, spec: ASRDatasetSpec, strict_attack_fields: bool = False) -> list[str]:
    errors: list[str] = []
    if not isinstance(item, dict):
        return [f"item is {type(item).__name__}, expected dict"]
    missing = CORE_FIELDS - set(item)
    if missing:
        errors.append(f"缺少核心字段: {sorted(missing)}")
    if strict_attack_fields and spec.attack == "ofa":
        missing_ofa = OFA_FIELDS - set(item)
        if missing_ofa:
            errors.append(f"缺少 OFA 字段: {sorted(missing_ofa)}")
    if not isinstance(item.get("Ans"), list):
        errors.append("Ans 不是 list")
    else:
        if item.get("totQA") is not None and len(item["Ans"]) not in {0, int(item.get("totQA", -1))}:
            errors.append(f"Ans 长度 {len(item['Ans'])} != totQA {item.get('totQA')}")
        for idx, ans in enumerate(item["Ans"][:3]):
            if not isinstance(ans, dict):
                errors.append(f"Ans[{idx}] 不是 dict")
                continue
            missing_ans = ANSWER_FIELDS - set(ans)
            if missing_ans:
                errors.append(f"Ans[{idx}] 缺少字段: {sorted(missing_ans)}")
    for field in ("tx", "txl", "totQA", "round"):
        if field in item and not isinstance(item[field], int):
            errors.append(f"{field} 类型为 {type(item[field]).__name__}，期望 int")
    if "questionID" in item:
        try:
            parse_question_id(item["questionID"])
        except Exception as exc:
            errors.append(f"questionID 无法解析: {item['questionID']!r} ({exc})")
    if isinstance(item.get("totQA"), int):
        for field in ("tx", "txl"):
            value = item.get(field)
            if isinstance(value, int) and not (0 <= value <= item["totQA"]):
                errors.append(f"{field}={value} 超出 [0, totQA={item['totQA']}]")
    return errors


def validate_file(path: str | Path, spec: ASRDatasetSpec, sample_items: int | None = None) -> dict[str, Any]:
    data = load_txfilled_file(path)
    items = data if sample_items is None else data[:sample_items]
    errors: list[str] = []
    if len(data) != spec.expected_records_per_file:
        errors.append(f"record_count={len(data)}，期望 {spec.expected_records_per_file}")
    for idx, item in enumerate(items):
        for err in validate_item(item, spec, strict_attack_fields=True):
            errors.append(f"item[{idx}]: {err}")
    keysets = sorted({tuple(sorted(item.keys())) for item in items if isinstance(item, dict)})
    return {"path": str(path), "record_count": len(data), "checked_items": len(items), "keysets": keysets, "errors": errors}
