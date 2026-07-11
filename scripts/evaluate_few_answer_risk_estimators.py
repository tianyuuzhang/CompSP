#!/usr/bin/env python3
"""用少量回答估计独立后缀 ASR，比较不同黑盒风险估计器。

每条指令的 20 个回答按确定性随机划分为 k 条前缀与未使用后缀。估计器只读取前缀，
目标只由后缀回答计算，避免“用参与目标统计的同一回答预测目标”的直接泄漏。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from compsp.asr_dataset import iter_records
from compsp.response_structure import pairwise_accuracy_by_group, pairwise_accuracy_by_group_macro
from compsp.safety_metrics import derive_alternative_metrics
from compsp.whitebox.supervised_direction import TEST_QIDS, TRAIN_QIDS, parse_csv_ints


def answer_is_unsafe(answer: dict) -> bool:
    return "unsafe" in str(answer.get("toxic", "")).lower()


def answer_is_long(answer: dict, threshold: int) -> bool:
    value = answer.get("len")
    if isinstance(value, (int, float)):
        return value >= threshold
    return len(str(answer.get("A", ""))) >= threshold


def safe_corr(left: np.ndarray, right: np.ndarray, method: str) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3 or np.std(left[mask]) == 0 or np.std(right[mask]) == 0:
        return float("nan")
    result = pearsonr(left[mask], right[mask]) if method == "pearson" else spearmanr(left[mask], right[mask])
    return float(result.statistic)


def evaluate(rows: list[dict], pred_key: str) -> dict:
    y = np.asarray([row["future_asr"] for row in rows], dtype=float)
    pred = np.asarray([row[pred_key] for row in rows], dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    usable = [row for row, keep in zip(rows, mask.tolist()) if keep]
    groups = [(row["dataset_key"], row["question_id"]) for row in usable]
    return {
        "n": int(mask.sum()),
        "pearson": safe_corr(pred, y, "pearson"),
        "spearman": safe_corr(pred, y, "spearman"),
        "mae": float(np.mean(np.abs(pred[mask] - y[mask]))),
        "brier": float(np.mean((pred[mask] - y[mask]) ** 2)),
        "同题序关系准确率": pairwise_accuracy_by_group(y[mask], pred[mask], groups),
        "按问题宏平均序准确率": pairwise_accuracy_by_group_macro(y[mask], pred[mask], groups),
    }


def aggregate_seed_metrics(seed_reports: list[dict]) -> dict:
    keys = seed_reports[0].keys()
    return {
        key: {
            "mean": float(np.nanmean([report[key] for report in seed_reports])),
            "std": float(np.nanstd([report[key] for report in seed_reports])),
        }
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="评估少量回答对独立后缀 ASR 的预测能力。")
    parser.add_argument("--dataset-keys", required=True)
    parser.add_argument("--question-ids", default=",".join(map(str, sorted(TRAIN_QIDS + TEST_QIDS))))
    parser.add_argument("--sample-sizes", default="1,2,4,8,10")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--length-threshold", type=int, default=500)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_keys = [key.strip() for key in args.dataset_keys.split(",") if key.strip()]
    question_ids = parse_csv_ints(args.question_ids, sorted(TRAIN_QIDS + TEST_QIDS))
    sample_sizes = [int(value) for value in args.sample_sizes.split(",") if value.strip()]
    records = []
    for dataset_key in dataset_keys:
        for qid in question_ids:
            for item_index, record in enumerate(iter_records(dataset_key, [qid])):
                if len(record.answers) < 2:
                    continue
                records.append((dataset_key, qid, item_index, record.answers))

    methods = ("prefix_empirical_asr", "prefix_jeffreys_asr", "prefix_alr", "prefix_judge_hazard", "prefix_hazard_weighted_asr")
    report = {
        "说明": "前缀估计器只读取 k 条回答；future_asr 只由未使用后缀回答计算。",
        "datasets": dataset_keys,
        "records": len(records),
        "seeds": args.seeds,
        "sample_sizes": {},
    }
    for k in sample_sizes:
        seed_results = []
        for seed in range(args.seeds):
            rows = []
            for dataset_key, qid, item_index, answers in records:
                if k >= len(answers):
                    continue
                local_seed = seed * 1_000_003 + qid * 1009 + item_index * 917 + sum(map(ord, dataset_key))
                order = np.random.default_rng(local_seed).permutation(len(answers))
                prefix = [answers[int(index)] for index in order[:k]]
                suffix = [answers[int(index)] for index in order[k:]]
                prefix_tx = sum(answer_is_unsafe(answer) for answer in prefix)
                prefix_txl = sum(answer_is_long(answer, args.length_threshold) for answer in prefix)
                future_tx = sum(answer_is_unsafe(answer) for answer in suffix)
                metrics = derive_alternative_metrics(prefix, prefix_tx, prefix_txl, len(prefix))
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "question_id": qid,
                        "future_asr": future_tx / len(suffix),
                        "prefix_empirical_asr": metrics.empirical_asr,
                        "prefix_jeffreys_asr": metrics.jeffreys_asr,
                        "prefix_alr": prefix_txl / len(prefix),
                        "prefix_judge_hazard": metrics.judge_hazard,
                        "prefix_hazard_weighted_asr": metrics.hazard_weighted_asr,
                    }
                )
            seed_report = {"合并": {}, "按数据集": {}}
            for method in methods:
                seed_report["合并"][method] = evaluate(rows, method)
                for dataset_key in dataset_keys:
                    subset = [row for row in rows if row["dataset_key"] == dataset_key]
                    seed_report["按数据集"].setdefault(dataset_key, {})[method] = evaluate(subset, method)
            seed_results.append(seed_report)

        k_report = {"合并": {}, "按数据集": {}}
        for method in methods:
            k_report["合并"][method] = aggregate_seed_metrics(
                [seed_report["合并"][method] for seed_report in seed_results]
            )
            for dataset_key in dataset_keys:
                k_report["按数据集"].setdefault(dataset_key, {})[method] = aggregate_seed_metrics(
                    [seed_report["按数据集"][dataset_key][method] for seed_report in seed_results]
                )
        report["sample_sizes"][str(k)] = k_report
        print(f"完成 k={k}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入 {output}")


if __name__ == "__main__":
    main()
