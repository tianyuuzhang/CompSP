#!/usr/bin/env python3
"""用少量回答文本预测独立后缀 ASR，避免同样本标签泄漏。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from compsp.asr_dataset import iter_records
from compsp.response_structure import answer_view, pairwise_accuracy_by_group, pairwise_accuracy_by_group_macro
from compsp.whitebox.supervised_direction import TEST_QIDS, TRAIN_QIDS, parse_csv_ints


def answer_is_unsafe(answer: dict) -> bool:
    return "unsafe" in str(answer.get("toxic", "")).lower()


def safe_corr(left: np.ndarray, right: np.ndarray, method: str) -> float:
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    result = pearsonr(left, right) if method == "pearson" else spearmanr(left, right)
    return float(result.statistic)


def metrics(rows: list[dict], pred: np.ndarray) -> dict:
    y = np.asarray([row["future_asr"] for row in rows], dtype=float)
    groups = [(row["dataset_key"], row["question_id"]) for row in rows]
    return {
        "n": len(rows),
        "pearson": safe_corr(pred, y, "pearson"),
        "spearman": safe_corr(pred, y, "spearman"),
        "mae": float(np.mean(np.abs(pred - y))),
        "brier": float(np.mean((pred - y) ** 2)),
        "同题序关系准确率": pairwise_accuracy_by_group(y, pred, groups),
        "按问题宏平均序准确率": pairwise_accuracy_by_group_macro(y, pred, groups),
    }


def aggregate(reports: list[dict]) -> dict:
    return {
        key: {"mean": float(np.nanmean([report[key] for report in reports])), "std": float(np.nanstd([report[key] for report in reports]))}
        for key in reports[0]
    }


def build_rows(records: list[tuple], k: int, seed: int) -> list[dict]:
    rows = []
    for dataset_key, qid, item_index, answers in records:
        if len(answers) <= k:
            continue
        local_seed = seed * 1_000_003 + qid * 1009 + item_index * 917 + sum(map(ord, dataset_key))
        order = np.random.default_rng(local_seed).permutation(len(answers))
        prefix = [answers[int(index)] for index in order[:k]]
        suffix = [answers[int(index)] for index in order[k:]]
        text = "\n\n<回答分隔>\n\n".join(answer_view(str(answer.get("A", ""))).substantive_text for answer in prefix)
        rows.append(
            {
                "dataset_key": dataset_key,
                "question_id": qid,
                "split": "train" if qid in TRAIN_QIDS else "test",
                "text": text[:12000],
                "future_asr": sum(answer_is_unsafe(answer) for answer in suffix) / len(suffix),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="少量回答文本对独立后缀 ASR 的严格预测检验。")
    parser.add_argument("--dataset-keys", required=True)
    parser.add_argument("--question-ids", default=",".join(map(str, sorted(TRAIN_QIDS + TEST_QIDS))))
    parser.add_argument("--sample-sizes", default="1,4")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-features", type=int, default=30000)
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

    report = {"说明": "TF-IDF 只读取前缀回答文本；目标是未使用后缀回答计算的 ASR。", "records": len(records), "results": {}}
    for k in [int(value) for value in args.sample_sizes.split(",") if value.strip()]:
        seed_reports = []
        for seed in range(args.seeds):
            rows = build_rows(records, k, seed)
            train = [row for row in rows if row["split"] == "train"]
            test = [row for row in rows if row["split"] == "test"]
            vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=4, max_features=args.max_features, sublinear_tf=True
            )
            x_train = vectorizer.fit_transform([row["text"] for row in train])
            x_test = vectorizer.transform([row["text"] for row in test])
            y_train = np.asarray([row["future_asr"] for row in train], dtype=float)
            pred = Ridge(alpha=10.0, solver="lsqr").fit(x_train, y_train).predict(x_test)
            seed_report = {"合并": metrics(test, pred), "按数据集": {}}
            for dataset_key in dataset_keys:
                mask = np.asarray([row["dataset_key"] == dataset_key for row in test])
                seed_report["按数据集"][dataset_key] = metrics(
                    [row for row, keep in zip(test, mask.tolist()) if keep], pred[mask]
                )
            seed_reports.append(seed_report)
            print(f"完成 k={k}, seed={seed}", flush=True)
        report["results"][str(k)] = {"合并": aggregate([item["合并"] for item in seed_reports]), "按数据集": {}}
        for dataset_key in dataset_keys:
            report["results"][str(k)]["按数据集"][dataset_key] = aggregate(
                [item["按数据集"][dataset_key] for item in seed_reports]
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入 {output}")


if __name__ == "__main__":
    main()
