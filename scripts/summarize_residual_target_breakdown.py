#!/usr/bin/env python3
"""按测试目标模型汇总回答残差迁移矩阵。

跨模型矩阵的整体平均会掩盖目标模型差异。本脚本对每个测试目标模型，平均四个训练来源的
同题序准确率，并输出 response 相对长度/格式基线的增益、基线+文本残差相对基线的增益。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


METRICS = ("asr", "hazard_weighted_asr", "alr")
METHOD_BASE = "仅长度格式基线"
METHOD_RESPONSE = "仅response文本"
METHOD_COMBO = "长度格式基线_plus_文本残差"
METHOD_RESID = "残差本身"


def summarize_one(label: str, path: Path) -> dict:
    """读取一个残差矩阵汇总文件，按测试目标模型聚合。"""

    data = json.loads(path.read_text(encoding="utf-8"))
    output = {"label": label, "path": str(path), "targets": {}}
    for metric in METRICS:
        output["targets"][metric] = {}
        for target in data["targets"]:
            base_values: list[float] = []
            response_values: list[float] = []
            combo_values: list[float] = []
            residual_values: list[float] = []
            for source in data["sources"]:
                cell = data["cells"][f"{source}_to_{target}"][metric]
                base_values.append(cell[METHOD_BASE]["同题序关系准确率"])
                response_values.append(cell[METHOD_RESPONSE]["同题序关系准确率"])
                combo_values.append(cell[METHOD_COMBO]["同题序关系准确率"])
                residual_values.append(cell[METHOD_RESID]["同题序关系准确率"])
            base = statistics.mean(base_values)
            response = statistics.mean(response_values)
            combo = statistics.mean(combo_values)
            residual = statistics.mean(residual_values)
            output["targets"][metric][target] = {
                "长度格式基线": base,
                "仅response文本": response,
                "长度格式基线_plus_文本残差": combo,
                "残差本身": residual,
                "response_minus_base": response - base,
                "combo_minus_base": combo - base,
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="按测试目标模型汇总残差矩阵增益。")
    parser.add_argument("--jbb", required=True)
    parser.add_argument("--hb", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    reports = [
        summarize_one("JBB", Path(args.jbb)),
        summarize_one("HarmBench", Path(args.hb)),
    ]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "residual_target_breakdown.json"
    json_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 回答残差矩阵的目标模型分层", ""]
    lines.append("每格为对四个训练来源取平均后的同题序准确率或增益。")
    lines.append("")
    for report in reports:
        lines.append(f"## {report['label']}")
        lines.append("")
        for metric in METRICS:
            lines.append(f"### {metric}")
            lines.extend([
                "",
                "| 测试目标 | 长度格式基线 | response 增益 | 基线+残差增益 | 残差本身 |",
                "|---|---:|---:|---:|---:|",
            ])
            for target, values in report["targets"][metric].items():
                lines.append(
                    f"| {target} | {values['长度格式基线']:.3f} | "
                    f"{values['response_minus_base']:.3f} | "
                    f"{values['combo_minus_base']:.3f} | "
                    f"{values['残差本身']:.3f} |"
                )
            lines.append("")
    lines.extend([
        "解释：`response 增益` 是仅回答文本相对长度/格式基线的提升；`基线+残差增益` 是两阶段残差模型相对基线的提升。",
        "若某个目标模型增益集中较强，说明跨模型平均可能由该目标模型驱动，需要单独解释。",
    ])
    md_path = out_dir / "目标模型分层汇总.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入 {json_path}")
    print(f"已写入 {md_path}")


if __name__ == "__main__":
    main()
