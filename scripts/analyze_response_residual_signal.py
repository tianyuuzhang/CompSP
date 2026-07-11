#!/usr/bin/env python3
"""检验回答文本在扣除长度/格式特征后的增量信号。

流程：
1. 用训练集的长度/拒绝/格式手工特征预测目标，得到基线模型。
2. 用 response TF-IDF 预测训练集基线残差。
3. 在测试集上比较“仅基线”和“基线 + 文本残差”的同题序关系准确率。

该分析用于区分“回答文本信号只是长度/格式”与“回答具体内容仍有增量”。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from analyze_response_safety_structure import (
    FEATURE_NAMES,
    build_tfidf_matrices,
    load_rows,
    sliced_metrics,
)


FEATURE_SETS = {
    "长度拒绝格式": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "仅长度": [0, 1, 4, 5],
}


def select_features(rows: list[dict], indices: list[int]) -> np.ndarray:
    """从聚合特征的均值/标准差/最小/最大四组中选取指定基础特征。"""

    base = len(FEATURE_NAMES)
    selected = [offset * base + index for offset in range(4) for index in indices]
    return np.stack([row["features"][selected] for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 response TF-IDF 的长度/格式残差信号。")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=1)
    parser.add_argument("--targets", default="pseudo_score,asr,alr")
    parser.add_argument("--train-datasets", default=None)
    parser.add_argument("--test-datasets", default=None)
    parser.add_argument("--text-cleaning", default="mask_refusal_hazard_terms")
    parser.add_argument("--feature-set", default="长度拒绝格式", choices=tuple(FEATURE_SETS))
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--alpha-base", type=float, default=10.0)
    parser.add_argument("--alpha-text", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    train_datasets = {x.strip() for x in args.train_datasets.split(",") if x.strip()} if args.train_datasets else None
    test_datasets = {x.strip() for x in args.test_datasets.split(",") if x.strip()} if args.test_datasets else None
    dataset_filter = (train_datasets or set()) | (test_datasets or set()) or None
    rows = load_rows(args.scores, args.sample_size, args.seed, None, targets, dataset_filter)
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

    x_base_train = select_features(train, FEATURE_SETS[args.feature_set])
    x_base_test = select_features(test, FEATURE_SETS[args.feature_set])
    scaler = StandardScaler().fit(x_base_train)
    xb_train = scaler.transform(x_base_train)
    xb_test = scaler.transform(x_base_test)
    x_text_train, x_text_test = build_tfidf_matrices(train, test, args.max_features, "response", args.text_cleaning)
    groups = [(row["dataset_key"], row["question_id"]) for row in test]

    report = {
        "说明": "基线先使用长度/拒绝/格式手工特征；文本模型只拟合训练残差。",
        "scores": args.scores,
        "sample_size": args.sample_size,
        "targets": targets,
        "text_cleaning": args.text_cleaning,
        "feature_set": args.feature_set,
        "train_datasets": sorted(train_datasets) if train_datasets else "全部",
        "test_datasets": sorted(test_datasets) if test_datasets else "全部",
        "train_records": len(train),
        "test_records": len(test),
        "results": {},
    }

    for target in targets:
        y_train = np.asarray([row[target] for row in train])
        y_test = np.asarray([row[target] for row in test])
        base_model = Ridge(alpha=args.alpha_base).fit(xb_train, y_train)
        base_train = base_model.predict(xb_train)
        base_test = base_model.predict(xb_test)
        residual_model = Ridge(alpha=args.alpha_text, solver="lsqr").fit(x_text_train, y_train - base_train)
        residual_test = residual_model.predict(x_text_test)
        combined_test = base_test + residual_test
        full_text = Ridge(alpha=args.alpha_text, solver="lsqr").fit(x_text_train, y_train).predict(x_text_test)
        report["results"][target] = {
            "仅长度格式基线": sliced_metrics(test, y_test, base_test, groups),
            "仅response文本": sliced_metrics(test, y_test, full_text, groups),
            "长度格式基线_plus_文本残差": sliced_metrics(test, y_test, combined_test, groups),
            "残差本身": sliced_metrics(test, y_test - base_test, residual_test, groups),
        }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "response_residual_signal_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Response 文本的长度/格式残差信号", ""]
    lines.append(f"基线特征集：`{args.feature_set}`；文本清理：`{args.text_cleaning}`。")
    lines.append("")
    for target in targets:
        lines.append(f"## {target}")
        lines.extend(["", "| 模型 | 同题序准确率 | Spearman | R2 |", "|---|---:|---:|---:|"])
        for name, result in report["results"][target].items():
            merged = result["合并"]
            lines.append(
                f"| {name} | {merged['同题序关系准确率']:.3f} | "
                f"{merged['spearman']:.3f} | {merged['r2']:.3f} |"
            )
        lines.append("")
    md_path = out_dir / "response_residual_signal_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {json_path}")
    print(f"已写入 {md_path}")


if __name__ == "__main__":
    main()
