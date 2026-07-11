#!/usr/bin/env python3
"""在回答长度相近的 pair 上评估 response 嗅探能力。

常规同题序关系准确率会混入“哪个回答更长/更像拒绝”的线索。该脚本只保留同一
攻击方案、同一问题内实质内容长度差不超过阈值的 pair，检验文本模型在长度匹配后
是否仍能预测 ASR/ALR/伪安全投影的相对顺序。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from analyze_response_safety_structure import build_tfidf_matrices, load_rows


def length_matched_accuracy(
    rows: list[dict],
    targets: np.ndarray,
    predictions: np.ndarray,
    lengths: np.ndarray,
    max_length_delta: float,
    min_target_delta: float,
) -> dict[str, float]:
    """计算同题内、长度差受限的 pairwise accuracy。"""

    grouped: dict[tuple[str, int], list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault((row["dataset_key"], row["question_id"]), []).append(index)
    correct = 0.0
    total = 0
    per_group = []
    for indices in grouped.values():
        group_correct = 0.0
        group_total = 0
        for pos, left in enumerate(indices):
            for right in indices[pos + 1 :]:
                if abs(float(lengths[left] - lengths[right])) > max_length_delta:
                    continue
                delta = float(targets[left] - targets[right])
                if abs(delta) <= min_target_delta:
                    continue
                pred_delta = float(predictions[left] - predictions[right])
                if pred_delta == 0:
                    group_correct += 0.5
                elif (pred_delta > 0) == (delta > 0):
                    group_correct += 1.0
                group_total += 1
        if group_total:
            correct += group_correct
            total += group_total
            per_group.append(group_correct / group_total)
    return {
        "pair数": int(total),
        "同题序准确率": float(correct / total) if total else float("nan"),
        "按问题宏平均序准确率": float(np.mean(per_group)) if per_group else float("nan"),
        "有效问题数": int(len(per_group)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="长度匹配 pair 上的 response 嗅探评估。")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=1)
    parser.add_argument("--targets", default="pseudo_score,asr,alr")
    parser.add_argument("--train-datasets", default=None)
    parser.add_argument("--test-datasets", default=None)
    parser.add_argument("--text-cleaning", default="mask_refusal_hazard_terms")
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--length-deltas", default="100,250,500,1000")
    parser.add_argument("--min-target-delta", type=float, default=0.0)
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

    x_train, x_test = build_tfidf_matrices(train, test, args.max_features, "response", args.text_cleaning)
    lengths = np.asarray([row["features"][4] for row in test], dtype=float)
    deltas = [float(item.strip()) for item in args.length_deltas.split(",") if item.strip()]
    report = {
        "说明": "只保留同攻击方案、同问题内实质内容长度差不超过阈值的 pair。",
        "scores": args.scores,
        "sample_size": args.sample_size,
        "targets": targets,
        "text_cleaning": args.text_cleaning,
        "length_deltas": deltas,
        "min_target_delta": args.min_target_delta,
        "train_records": len(train),
        "test_records": len(test),
        "results": {},
    }
    for target in targets:
        y_train = np.asarray([row[target] for row in train])
        y_test = np.asarray([row[target] for row in test])
        pred = Ridge(alpha=args.alpha, solver="lsqr").fit(x_train, y_train).predict(x_test)
        target_result = {}
        for delta in deltas:
            target_result[f"长度差<={delta:g}"] = length_matched_accuracy(
                test, y_test, pred, lengths, delta, args.min_target_delta
            )
        report["results"][target] = target_result

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "length_matched_pairs_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 长度匹配 pair 上的回答文本嗅探", ""]
    lines.append(f"测试记录数：{len(test)}；文本清理：`{args.text_cleaning}`。")
    lines.append("")
    for target in targets:
        lines.append(f"## {target}")
        lines.extend(["", "| 长度阈值 | pair数 | 有效问题数 | 同题序准确率 | 宏平均序准确率 |", "|---|---:|---:|---:|---:|"])
        for name, item in report["results"][target].items():
            lines.append(
                f"| {name} | {item['pair数']} | {item['有效问题数']} | "
                f"{item['同题序准确率']:.3f} | {item['按问题宏平均序准确率']:.3f} |"
            )
        lines.append("")
    md_path = out_dir / "length_matched_pairs_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {json_path}")
    print(f"已写入 {md_path}")


if __name__ == "__main__":
    main()
