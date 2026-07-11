#!/usr/bin/env python3
"""按实质内容长度分层评估回答文本嗅探信号。

训练阶段仍使用全部训练题；测试阶段按去拒绝前缀后的实质内容字符数分为短/中/长，
分别报告 TF-IDF response-only 对 ASR、ALR 和伪安全投影的预测效果。该脚本用于判断
信号是否主要来自“长回答”的具体内容，而不是只来自短拒绝或长度本身。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from analyze_response_safety_structure import (
    build_tfidf_matrices,
    load_rows,
    metrics,
    sliced_metrics,
)


def stratum_masks(lengths: np.ndarray) -> dict[str, np.ndarray]:
    """按测试集实质内容长度三分位生成短/中/长掩码。"""

    low, high = np.quantile(lengths, [1 / 3, 2 / 3])
    return {
        f"短实质内容_<= {low:.1f}": lengths <= low,
        f"中实质内容_({low:.1f}, {high:.1f}]": (lengths > low) & (lengths <= high),
        f"长实质内容_> {high:.1f}": lengths > high,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="按回答实质内容长度分层评估 TF-IDF 嗅探。")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=1)
    parser.add_argument("--targets", default="pseudo_score,asr,alr")
    parser.add_argument("--train-datasets", default=None)
    parser.add_argument("--test-datasets", default=None)
    parser.add_argument("--text-cleaning", default="mask_refusal_hazard_terms")
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    target_names = [item.strip() for item in args.targets.split(",") if item.strip()]
    train_datasets = {x.strip() for x in args.train_datasets.split(",") if x.strip()} if args.train_datasets else None
    test_datasets = {x.strip() for x in args.test_datasets.split(",") if x.strip()} if args.test_datasets else None
    dataset_filter = (train_datasets or set()) | (test_datasets or set()) or None
    rows = load_rows(args.scores, args.sample_size, args.seed, None, target_names, dataset_filter)
    train = [
        row for row in rows
        if row["split"] == "train" and (train_datasets is None or row["dataset_key"] in train_datasets)
    ]
    test = [
        row for row in rows
        if row["split"] == "test" and (test_datasets is None or row["dataset_key"] in test_datasets)
    ]
    if not train or not test:
        raise ValueError("训练集或测试集为空。")

    x_train, x_test = build_tfidf_matrices(train, test, args.max_features, "response", args.text_cleaning)
    groups = [(row["dataset_key"], row["question_id"]) for row in test]
    lengths = np.asarray([row["features"][4] for row in test], dtype=float)
    masks = stratum_masks(lengths)

    report = {
        "说明": "训练使用全部训练题；测试按实质内容字符数三分位分层。输入特征仅为回答文本 TF-IDF。",
        "scores": args.scores,
        "sample_size": args.sample_size,
        "targets": target_names,
        "text_cleaning": args.text_cleaning,
        "train_datasets": sorted(train_datasets) if train_datasets else "全部",
        "test_datasets": sorted(test_datasets) if test_datasets else "全部",
        "train_records": len(train),
        "test_records": len(test),
        "length_summary": {
            "min": float(np.min(lengths)),
            "q1": float(np.quantile(lengths, 0.25)),
            "median": float(np.median(lengths)),
            "q3": float(np.quantile(lengths, 0.75)),
            "max": float(np.max(lengths)),
            "mean": float(np.mean(lengths)),
        },
        "results": {},
    }

    for target in target_names:
        y_train = np.asarray([row[target] for row in train])
        y_test = np.asarray([row[target] for row in test])
        pred = Ridge(alpha=args.alpha, solver="lsqr").fit(x_train, y_train).predict(x_test)
        target_result = {"合并": sliced_metrics(test, y_test, pred, groups), "按长度分层": {}}
        for name, mask in masks.items():
            indices = np.flatnonzero(mask)
            subset_test = [test[int(i)] for i in indices]
            subset_groups = [groups[int(i)] for i in indices]
            target_result["按长度分层"][name] = {
                "记录数": int(mask.sum()),
                "平均实质内容字符数": float(np.mean(lengths[mask])),
                "评估": sliced_metrics(subset_test, y_test[mask], pred[mask], subset_groups),
            }
        report["results"][target] = target_result

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "response_length_strata_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 回答实质内容长度分层嗅探结果", ""]
    lines.append(f"样本数：训练 {len(train)}，测试 {len(test)}；清理模式：`{args.text_cleaning}`。")
    lines.append("")
    for target in target_names:
        lines.append(f"## {target}")
        merged = report["results"][target]["合并"]["合并"]
        lines.append(
            f"- 合并同题序准确率：{merged['同题序关系准确率']:.3f}；"
            f"Spearman：{merged['spearman']:.3f}"
        )
        lines.extend(["", "| 分层 | n | 平均实质内容字符数 | 同题序准确率 | Spearman |", "|---|---:|---:|---:|---:|"])
        for name, item in report["results"][target]["按长度分层"].items():
            m = item["评估"]["合并"]
            lines.append(
                f"| {name} | {item['记录数']} | {item['平均实质内容字符数']:.1f} | "
                f"{m['同题序关系准确率']:.3f} | {m['spearman']:.3f} |"
            )
        lines.append("")
    md_path = out_dir / "response_length_strata_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {json_path}")
    print(f"已写入 {md_path}")


if __name__ == "__main__":
    main()
