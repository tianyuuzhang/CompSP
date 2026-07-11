#!/usr/bin/env python3
"""把回答嗅探迁移矩阵的单元 JSON 汇总为中文 Markdown 和结构化 JSON。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("asr", "hazard_weighted_asr", "alr")


def read_cell(path: Path, target: str) -> dict | None:
    report_file = path / "response_safety_structure_report.json"
    if not report_file.exists():
        return None
    report = json.loads(report_file.read_text(encoding="utf-8"))
    try:
        target_results = report["results"]["1"]["目标"][target]
    except KeyError:
        return None
    for method_name, method_result in target_results.items():
        if method_name.startswith("TFIDF_Ridge_"):
            try:
                return method_result["评估"]["合并"]
            except KeyError:
                return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总回答文本迁移矩阵。")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    cells: dict[tuple[str, str], dict] = {}
    sources, targets = set(), set()
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or "_to_" not in directory.name:
            continue
        source, target = directory.name.split("_to_", 1)
        sources.add(source)
        targets.add(target)
        cells[(source, target)] = {metric: read_cell(directory, metric) for metric in METRICS}

    source_list = sorted(sources)
    target_list = sorted(targets)
    output = {
        "root": str(root),
        "sources": source_list,
        "targets": target_list,
        "cells": {f"{source}_to_{target}": value for (source, target), value in cells.items()},
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transfer_matrix_summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 回答嗅探迁移矩阵汇总", ""]
    for metric in METRICS:
        lines.extend([f"## {metric}", "", "| 训练来源 \\ 测试目标 | " + " | ".join(target_list) + " |", "|---|" + "---|" * len(target_list)])
        for source in source_list:
            values = []
            for target in target_list:
                entry = cells.get((source, target), {}).get(metric)
                if entry is None:
                    values.append("缺失")
                else:
                    values.append(f"{entry.get('同题序关系准确率', float('nan')):.3f}")
            lines.append("| " + source + " | " + " | ".join(values) + " |")
        lines.append("")
    lines.extend([
        "说明：每格用源域 60 个训练问题拟合 TF-IDF Ridge，在目标域 20 个留出问题评估。",
        "所有目标标签均来自测试目标模型自身；数值是同题内序关系准确率。",
    ])
    (out_dir / "迁移矩阵汇总.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入 {out_dir}")


if __name__ == "__main__":
    main()
