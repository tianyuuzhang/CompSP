#!/usr/bin/env python3
"""比较 response、遮蔽 response 与 q1-only 的迁移矩阵。

输出用于判断回答文本是否提供输入 q1 之外的增量信号，以及显式拒绝/危险词遮蔽后
这种增量是否仍然存在。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("asr", "hazard_weighted_asr", "alr")


def load_summary(root: Path) -> dict:
    path = root / "transfer_matrix_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def cell_value(summary: dict, cell: str, metric: str) -> float | None:
    entry = summary.get("cells", {}).get(cell, {}).get(metric)
    if not entry:
        return None
    value = entry.get("同题序关系准确率")
    return None if value is None else float(value)


def fmt(value: float | None) -> str:
    return "缺失" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="比较回答文本和 q1 baseline 的迁移矩阵。")
    parser.add_argument("--response-root", required=True)
    parser.add_argument("--masked-root", required=True)
    parser.add_argument("--q1-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    response = load_summary(Path(args.response_root))
    masked = load_summary(Path(args.masked_root))
    q1 = load_summary(Path(args.q1_root))
    sources = sorted(set(response["sources"]) | set(masked["sources"]) | set(q1["sources"]))
    targets = sorted(set(response["targets"]) | set(masked["targets"]) | set(q1["targets"]))

    rows = []
    for metric in METRICS:
        for source in sources:
            for target in targets:
                cell = f"{source}_to_{target}"
                response_value = cell_value(response, cell, metric)
                masked_value = cell_value(masked, cell, metric)
                q1_value = cell_value(q1, cell, metric)
                rows.append(
                    {
                        "metric": metric,
                        "source": source,
                        "target": target,
                        "response": response_value,
                        "masked_response": masked_value,
                        "q1": q1_value,
                        "masked_minus_response": None
                        if masked_value is None or response_value is None
                        else masked_value - response_value,
                        "masked_minus_q1": None
                        if masked_value is None or q1_value is None
                        else masked_value - q1_value,
                    }
                )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "response_q1_transfer_comparison.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = ["# Response 与 q1 迁移矩阵差值汇总", ""]
    lines.append("说明：`遮蔽-response - response` 反映显式拒绝/危险词遮蔽后的变化；")
    lines.append("`遮蔽-response - q1` 反映回答文本相对输入文本 baseline 的增量。")
    lines.append("")
    for metric in METRICS:
        metric_rows = [row for row in rows if row["metric"] == metric]
        avg_mask = [row["masked_minus_response"] for row in metric_rows if row["masked_minus_response"] is not None]
        avg_q1 = [row["masked_minus_q1"] for row in metric_rows if row["masked_minus_q1"] is not None]
        lines.extend(
            [
                f"## {metric}",
                "",
                f"- 平均 `遮蔽-response - response`：{sum(avg_mask) / len(avg_mask):.4f}",
                f"- 平均 `遮蔽-response - q1`：{sum(avg_q1) / len(avg_q1):.4f}",
                "",
                "| 训练来源 | 测试目标 | response | 遮蔽 response | q1 | 遮蔽-response - response | 遮蔽-response - q1 |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in metric_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["source"],
                        row["target"],
                        fmt(row["response"]),
                        fmt(row["masked_response"]),
                        fmt(row["q1"]),
                        fmt(row["masked_minus_response"]),
                        fmt(row["masked_minus_q1"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    output = out_dir / "response_q1_transfer_comparison.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {output}")


if __name__ == "__main__":
    main()
