#!/usr/bin/env python3
"""离线模拟基于少量回答的序贯 ASR 增采策略。

每条指令的 20 个回答被随机拆成前缀和后缀。策略只看前 k 条回答形成的估计器，
在同一数据集、同一问题内选择 top fraction 指令作为“追加采样候选”，再用未使用后缀
ASR 评估这些候选是否确实更高风险。该模拟不读取 toxicA 细节，只用回答级 unsafe/长度。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compsp.asr_dataset import iter_records
from compsp.safety_metrics import derive_alternative_metrics
from compsp.whitebox.supervised_direction import TEST_QIDS, TRAIN_QIDS, parse_csv_ints


def answer_is_unsafe(answer: dict) -> bool:
    """回答级 unsafe 判定。"""

    return "unsafe" in str(answer.get("toxic", "")).lower()


def answer_is_long(answer: dict, threshold: int) -> bool:
    """回答长度判定，用于前缀 ALR。"""

    value = answer.get("len")
    if isinstance(value, (int, float)):
        return value >= threshold
    return len(str(answer.get("A", ""))) >= threshold


def build_rows(records: list[tuple], k: int, seed: int, length_threshold: int) -> list[dict]:
    """构造一次随机前缀/后缀拆分。"""

    rows = []
    for dataset_key, qid, item_index, answers in records:
        if len(answers) <= k:
            continue
        local_seed = seed * 1_000_003 + qid * 1009 + item_index * 917 + sum(map(ord, dataset_key))
        order = np.random.default_rng(local_seed).permutation(len(answers))
        prefix = [answers[int(index)] for index in order[:k]]
        suffix = [answers[int(index)] for index in order[k:]]
        prefix_tx = sum(answer_is_unsafe(answer) for answer in prefix)
        prefix_txl = sum(answer_is_long(answer, length_threshold) for answer in prefix)
        future_tx = sum(answer_is_unsafe(answer) for answer in suffix)
        metrics = derive_alternative_metrics(prefix, prefix_tx, prefix_txl, len(prefix))
        rows.append(
            {
                "dataset_key": dataset_key,
                "question_id": qid,
                "item_index": item_index,
                "future_asr": future_tx / len(suffix),
                "prefix_empirical_asr": metrics.empirical_asr,
                "prefix_jeffreys_asr": metrics.jeffreys_asr,
                "prefix_alr": prefix_txl / len(prefix),
                "prefix_judge_hazard": metrics.judge_hazard,
                "prefix_hazard_weighted_asr": metrics.hazard_weighted_asr,
                "prefix_zero": prefix_tx == 0,
            }
        )
    return rows


def select_within_groups(rows: list[dict], score_key: str, top_fraction: float, rng: np.random.Generator) -> list[dict]:
    """在同数据集、同问题内选择分数最高的一部分。"""

    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["dataset_key"], row["question_id"]), []).append(row)
    selected = []
    for group_rows in groups.values():
        count = max(1, int(round(len(group_rows) * top_fraction)))
        if score_key == "random":
            order = rng.permutation(len(group_rows))[:count]
            selected.extend(group_rows[int(index)] for index in order)
            continue
        ranked = sorted(group_rows, key=lambda item: (item[score_key], rng.random()), reverse=True)
        selected.extend(ranked[:count])
    return selected


def evaluate_selection(rows: list[dict], selected: list[dict], high_threshold: float) -> dict:
    """评估被选择集合的未来 ASR 和高风险召回。"""

    selected_ids = {(row["dataset_key"], row["question_id"], row["item_index"]) for row in selected}
    high_all = [row for row in rows if row["future_asr"] >= high_threshold]
    high_selected = [row for row in selected if row["future_asr"] >= high_threshold]
    zero_rows = [row for row in rows if row["prefix_zero"]]
    zero_selected = [row for row in selected if row["prefix_zero"]]
    hidden_high = [row for row in zero_rows if row["future_asr"] >= high_threshold]
    hidden_high_selected = [row for row in zero_selected if row["future_asr"] >= high_threshold]
    return {
        "selected_n": len(selected),
        "selected_fraction": len(selected) / len(rows) if rows else float("nan"),
        "future_asr_mean_all": float(np.mean([row["future_asr"] for row in rows])),
        "future_asr_mean_selected": float(np.mean([row["future_asr"] for row in selected])),
        "future_asr_lift": float(np.mean([row["future_asr"] for row in selected]) - np.mean([row["future_asr"] for row in rows])),
        "high_threshold": high_threshold,
        "high_precision": len(high_selected) / len(selected) if selected else float("nan"),
        "high_recall": len(high_selected) / len(high_all) if high_all else float("nan"),
        "hidden_high_prefix_zero_total": len(hidden_high),
        "hidden_high_prefix_zero_selected": len(hidden_high_selected),
        "hidden_high_prefix_zero_recall": len(hidden_high_selected) / len(hidden_high) if hidden_high else float("nan"),
    }


def aggregate(reports: list[dict]) -> dict:
    """对多个 seed 的同名指标求均值和标准差。"""

    return {
        key: {
            "mean": float(np.nanmean([report[key] for report in reports])),
            "std": float(np.nanstd([report[key] for report in reports])),
        }
        for key in reports[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="模拟少量回答驱动的序贯 ASR 增采。")
    parser.add_argument("--dataset-keys", required=True)
    parser.add_argument("--question-ids", default=",".join(map(str, sorted(TRAIN_QIDS + TEST_QIDS))))
    parser.add_argument("--sample-sizes", default="1,2,4")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument("--high-threshold", type=float, default=0.5)
    parser.add_argument("--length-threshold", type=int, default=500)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_keys = [key.strip() for key in args.dataset_keys.split(",") if key.strip()]
    question_ids = parse_csv_ints(args.question_ids, sorted(TRAIN_QIDS + TEST_QIDS))
    records = []
    for dataset_key in dataset_keys:
        for qid in question_ids:
            for item_index, record in enumerate(iter_records(dataset_key, [qid])):
                if len(record.answers) > 1:
                    records.append((dataset_key, qid, item_index, record.answers))

    score_keys = [
        "random",
        "prefix_empirical_asr",
        "prefix_jeffreys_asr",
        "prefix_alr",
        "prefix_judge_hazard",
        "prefix_hazard_weighted_asr",
    ]
    report = {
        "说明": "前缀只用于选择追加采样候选；评估目标是未使用后缀 ASR。",
        "datasets": dataset_keys,
        "records": len(records),
        "top_fraction": args.top_fraction,
        "high_threshold": args.high_threshold,
        "sample_sizes": {},
    }
    for k in [int(value) for value in args.sample_sizes.split(",") if value.strip()]:
        seed_reports = {score: [] for score in score_keys}
        for seed in range(args.seeds):
            rows = build_rows(records, k, seed, args.length_threshold)
            rng = np.random.default_rng(seed + 20260712)
            for score in score_keys:
                selected = select_within_groups(rows, score, args.top_fraction, rng)
                seed_reports[score].append(evaluate_selection(rows, selected, args.high_threshold))
            print(f"完成 k={k}, seed={seed}", flush=True)
        report["sample_sizes"][str(k)] = {score: aggregate(values) for score, values in seed_reports.items()}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 序贯 ASR 增采离线模拟", ""]
    lines.append(f"选择比例：top {args.top_fraction:.0%}；高风险阈值：future ASR >= {args.high_threshold}。")
    lines.append("")
    for k, k_report in report["sample_sizes"].items():
        lines.extend([f"## 前缀回答数 k={k}", "", "| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |", "|---|---:|---:|---:|---:|---:|"])
        for score, values in k_report.items():
            lines.append(
                f"| {score} | {values['future_asr_mean_selected']['mean']:.3f} | "
                f"{values['future_asr_lift']['mean']:.3f} | "
                f"{values['high_precision']['mean']:.3f} | "
                f"{values['high_recall']['mean']:.3f} | "
                f"{values['hidden_high_prefix_zero_recall']['mean']:.3f} |"
            )
        lines.append("")
    md_path = output.with_suffix(".md")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"结果已写入 {output}")
    print(f"报告已写入 {md_path}")


if __name__ == "__main__":
    main()
