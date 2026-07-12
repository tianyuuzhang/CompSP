#!/usr/bin/env python3
"""汇总回答残差信号跨模型迁移矩阵。

输入目录由 `run_cross_model_ofa_response_residual_matrix.sh` 生成，每个
`source_to_target` 子目录包含 `response_residual_signal_report.json`。本脚本抽取
“仅长度格式基线”“仅response文本”“长度格式基线_plus_文本残差”“残差本身”
四类模型的同题序准确率，写成中文 Markdown 与结构化 JSON。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("asr", "hazard_weighted_asr", "alr")
MODELS = ("仅长度格式基线", "仅response文本", "长度格式基线_plus_文本残差", "残差本身")


def read_cell(path: Path, target: str, model_name: str) -> dict | None:
    """读取一个迁移矩阵单元中指定目标和模型的合并评估结果。"""

    report_file = path / "response_residual_signal_report.json"
    if not report_file.exists():
        return None
    report = json.loads(report_file.read_text(encoding="utf-8"))
    try:
        return report["results"][target][model_name]["合并"]
    except KeyError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总回答残差信号跨模型迁移矩阵。")
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
        cells[(source, target)] = {
            metric: {model: read_cell(directory, metric, model) for model in MODELS}
            for metric in METRICS
        }

    source_list = sorted(sources)
    target_list = sorted(targets)
    output = {
        "root": str(root),
        "sources": source_list,
        "targets": target_list,
        "metrics": list(METRICS),
        "models": list(MODELS),
        "cells": {f"{source}_to_{target}": value for (source, target), value in cells.items()},
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "residual_transfer_matrix_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = ["# 回答残差信号跨模型迁移矩阵", ""]
    lines.append("每格为同题内序关系准确率。基线使用长度、拒绝和格式特征；文本残差使用遮蔽拒绝词/危险词后的回答 TF-IDF。")
    lines.append("")
    for metric in METRICS:
        lines.append(f"## {metric}")
        lines.append("")
        for model_name in MODELS:
            lines.extend([
                f"### {model_name}",
                "",
                "| 训练来源 \\ 测试目标 | " + " | ".join(target_list) + " |",
                "|---|" + "---|" * len(target_list),
            ])
            for source in source_list:
                values = []
                for target in target_list:
                    entry = cells.get((source, target), {}).get(metric, {}).get(model_name)
                    if entry is None:
                        values.append("缺失")
                    else:
                        values.append(f"{entry.get('同题序关系准确率', float('nan')):.3f}")
                lines.append("| " + source + " | " + " | ".join(values) + " |")
            lines.append("")
    lines.extend([
        "说明：所有标签均来自测试目标模型自身；该实验不使用白盒 hidden state，也不使用安全判定字段作为输入。",
        "若“长度格式基线_plus_文本残差”稳定高于“仅长度格式基线”，说明回答文本中存在长度/格式之外的可迁移增量。",
    ])
    (out_dir / "残差迁移矩阵汇总.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入 {out_dir}")


if __name__ == "__main__":
    main()
