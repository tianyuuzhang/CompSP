#!/usr/bin/env python3
"""为一个或多个 ASR 数据集构建替代安全指标索引与相关性报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from compsp.asr_dataset import iter_records
from compsp.safety_metrics import derive_alternative_metrics
from compsp.whitebox.supervised_direction import TEST_QIDS, TRAIN_QIDS, parse_csv_ints, write_jsonl


METRIC_KEYS = (
    "empirical_asr",
    "jeffreys_asr",
    "judge_hazard",
    "first_vote_unsafe_rate",
    "delayed_unsafe_rate",
    "unanimous_three_safe_rate",
    "vote_error_rate",
    "alr",
)


def safe_corr(left: list[float], right: list[float], method: str) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float("nan")
    result = pearsonr(x[mask], y[mask]) if method == "pearson" else spearmanr(x[mask], y[mask])
    return float(result.statistic)


def summarize(rows: list[dict]) -> dict:
    report = {"rows": len(rows), "datasets": {}, "correlations_with_empirical_asr": {}}
    for dataset_key in sorted({row["dataset_key"] for row in rows}):
        items = [row for row in rows if row["dataset_key"] == dataset_key]
        report["datasets"][dataset_key] = {
            "rows": len(items),
            "questions": len({row["question_id"] for row in items}),
            "distinct_empirical_asr": len({row["empirical_asr"] for row in items}),
            "mean_valid_votes": float(np.mean([row["valid_vote_count"] for row in items])),
            "reconstruction_match_rate": float(
                np.mean([row["reconstructed_unsafe_answers"] == row["tx"] for row in items])
            ),
            "metrics": {},
        }
        empirical_items = [row["empirical_asr"] for row in items]
        for metric_key in METRIC_KEYS:
            values = np.asarray([row[metric_key] for row in items], dtype=float)
            finite = values[np.isfinite(values)]
            report["datasets"][dataset_key]["metrics"][metric_key] = {
                "finite_rows": int(len(finite)),
                "distinct_values": int(len(np.unique(finite))),
                "mean": float(np.mean(finite)) if len(finite) else float("nan"),
                "p05": float(np.quantile(finite, 0.05)) if len(finite) else float("nan"),
                "p50": float(np.quantile(finite, 0.50)) if len(finite) else float("nan"),
                "p95": float(np.quantile(finite, 0.95)) if len(finite) else float("nan"),
                "pearson_with_empirical_asr": safe_corr(empirical_items, values.tolist(), "pearson"),
                "spearman_with_empirical_asr": safe_corr(empirical_items, values.tolist(), "spearman"),
            }
    empirical = [row["empirical_asr"] for row in rows]
    for key in METRIC_KEYS[1:]:
        values = [row[key] for row in rows]
        report["correlations_with_empirical_asr"][key] = {
            "pearson": safe_corr(empirical, values, "pearson"),
            "spearman": safe_corr(empirical, values, "spearman"),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="构建提前停止判定流程下的替代 ASR 指标。")
    parser.add_argument("--dataset-keys", required=True, help="逗号分隔的数据集 key。")
    parser.add_argument("--question-ids", default=",".join(map(str, sorted(TRAIN_QIDS + TEST_QIDS))))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    dataset_keys = [key.strip() for key in args.dataset_keys.split(",") if key.strip()]
    question_ids = parse_csv_ints(args.question_ids, sorted(TRAIN_QIDS + TEST_QIDS))
    split_by_qid = {qid: "train" for qid in TRAIN_QIDS} | {qid: "test" for qid in TEST_QIDS}
    rows = []
    for dataset_key in dataset_keys:
        for qid in question_ids:
            for item_index, record in enumerate(iter_records(dataset_key, [qid])):
                derived = derive_alternative_metrics(record.answers, record.tx, record.tot_qa)
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "question_id": record.question_id,
                        "item_index": item_index,
                        "round": record.round,
                        "split": split_by_qid.get(record.question_id, "other"),
                        "q1": record.q1,
                        "tx": record.tx,
                        "txl": record.txl,
                        "tot_qa": record.tot_qa,
                        "alr": record.alr,
                        **derived.to_dict(),
                    }
                )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "alternative_asr_scores.jsonl", rows)
    report = summarize(rows)
    (out_dir / "alternative_asr_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
