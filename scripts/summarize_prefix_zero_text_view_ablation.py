#!/usr/bin/env python3
"""汇总 prefix-zero 隐蔽高风险模型的 response/q1/joint 三视图对照。

该脚本只读取已经生成的轻量 JSON 结果，不重新访问大数据。主要用途是判断
回答文本是否提供了相对原始攻击指令 q1 的增量信号，并生成可直接落档的中文
Markdown 表格。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_cell(report: dict, k: str, strategy: str) -> dict:
    result = report["sample_sizes"][k]["strategies"][strategy]
    top = result["top_fraction"]
    return {
        "roc_auc": float(result["roc_auc"]),
        "ap": float(result["average_precision"]),
        "top_precision": float(top["precision"]),
        "top_recall": float(top["recall"]),
        "max_f1": float(result["max_f1"]),
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 prefix-zero 三视图对照。")
    parser.add_argument("--input-dir", required=True, help="三视图 JSON 所在目录。")
    parser.add_argument("--prefix", default="llama_three_attack_prefix_zero_hidden_risk")
    parser.add_argument("--suffix", default="fast")
    parser.add_argument("--strategy", default="text_ridge", choices=("text_ridge", "text_logistic"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    reports = {}
    for view in ("response", "q1", "joint"):
        path = input_dir / f"{args.prefix}_{view}_{args.suffix}.json"
        if not path.exists():
            raise FileNotFoundError(f"缺少三视图结果: {path}")
        reports[view] = load_report(path)

    sample_sizes = sorted(reports["response"]["sample_sizes"], key=lambda value: int(value))
    summary = {
        "说明": "prefix-zero 三视图对照汇总；主比较为 response 相对 q1 的增量。",
        "strategy": args.strategy,
        "views": ["response", "q1", "joint"],
        "sample_sizes": {},
    }
    for k in sample_sizes:
        cells = {view: metric_cell(report, k, args.strategy) for view, report in reports.items()}
        summary["sample_sizes"][k] = {
            "views": cells,
            "response_minus_q1": {
                "ap": cells["response"]["ap"] - cells["q1"]["ap"],
                "top_recall": cells["response"]["top_recall"] - cells["q1"]["top_recall"],
                "roc_auc": cells["response"]["roc_auc"] - cells["q1"]["roc_auc"],
            },
            "joint_minus_best_single": {
                "ap": cells["joint"]["ap"] - max(cells["response"]["ap"], cells["q1"]["ap"]),
                "top_recall": cells["joint"]["top_recall"] - max(cells["response"]["top_recall"], cells["q1"]["top_recall"]),
                "roc_auc": cells["joint"]["roc_auc"] - max(cells["response"]["roc_auc"], cells["q1"]["roc_auc"]),
            },
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Prefix-zero 三视图对照汇总", ""]
    lines.append(f"主策略：`{args.strategy}`。")
    lines.append("")
    lines.append(
        "该表比较 `response`、`q1` 和 `joint` 三种输入视图。"
        "`response-q1` 为正时，说明回答文本相对原始攻击指令提供额外信号；"
        "`joint-best` 为正时，说明联合视图超过最强单视图。"
    )
    lines.append("")
    lines.extend(
        [
            "| k | 视图 | ROC-AUC | AP | top precision | top recall | max F1 |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for k in sample_sizes:
        for view in ("response", "q1", "joint"):
            cell = summary["sample_sizes"][k]["views"][view]
            lines.append(
                f"| {k} | {view} | {fmt(cell['roc_auc'])} | {fmt(cell['ap'])} | "
                f"{fmt(cell['top_precision'])} | {fmt(cell['top_recall'])} | {fmt(cell['max_f1'])} |"
            )
    lines.append("")
    lines.extend(
        [
            "| k | response-q1 AP | response-q1 top recall | joint-best AP | joint-best top recall |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for k in sample_sizes:
        diff = summary["sample_sizes"][k]["response_minus_q1"]
        joint = summary["sample_sizes"][k]["joint_minus_best_single"]
        lines.append(
            f"| {k} | {fmt(diff['ap'])} | {fmt(diff['top_recall'])} | "
            f"{fmt(joint['ap'])} | {fmt(joint['top_recall'])} |"
        )
    lines.append("")
    lines.append("解释口径：若 `q1` 接近或超过 `response`，应优先解释为输入攻击文本或主题结构可预测隐蔽风险；若 `response` 稳定超过 `q1`，才支持回答内容提供额外黑盒嗅探信号。")

    md_path = output.with_suffix(".md")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON 已写入 {output}")
    print(f"Markdown 已写入 {md_path}")


if __name__ == "__main__":
    main()
