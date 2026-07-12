#!/usr/bin/env python3
"""分析经验 ASR 的分辨率和低风险区分限制。

当前 txfilled 数据通常每条指令只有 20 次回答，因此经验 ASR 的步长是 0.05。
本脚本基于已构建的 alternative_asr_scores.jsonl，统计不同数据集的 ASR 档位、
Beta/Jeffreys 后验区间宽度，以及低 ASR 区间的样本比例。它不读取回答正文。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta


def interval_width(tx: int, total: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Jeffreys Beta(0.5,0.5) 后验的双侧可信区间。"""

    low = float(beta.ppf(alpha / 2, tx + 0.5, total - tx + 0.5))
    high = float(beta.ppf(1 - alpha / 2, tx + 0.5, total - tx + 0.5))
    return low, high, high - low


def summarize_rows(rows: list[dict]) -> dict:
    """汇总一个数据集或数据集组的 ASR 分辨率。"""

    tx_values = np.asarray([int(row["tx"]) for row in rows], dtype=int)
    totals = np.asarray([int(row["tot_qa"]) for row in rows], dtype=int)
    asr = tx_values / totals
    widths = np.asarray([interval_width(int(tx), int(total))[2] for tx, total in zip(tx_values, totals)])
    zero_high = np.asarray([interval_width(0, int(total))[1] for total in totals[tx_values == 0]])
    one_low = np.asarray([interval_width(int(total), int(total))[0] for total in totals[tx_values == totals]])
    unique_asr = sorted(set(round(float(value), 6) for value in asr))
    return {
        "n": len(rows),
        "totQA_values": sorted(set(int(value) for value in totals)),
        "unique_asr_count": len(unique_asr),
        "min_nonzero_step": float(min((value for value in unique_asr if value > 0), default=float("nan"))),
        "asr_mean": float(np.mean(asr)),
        "asr_std": float(np.std(asr)),
        "zero_asr_fraction": float(np.mean(tx_values == 0)),
        "one_asr_fraction": float(np.mean(tx_values == totals)),
        "low_asr_le_0_05_fraction": float(np.mean(asr <= 0.05)),
        "low_asr_le_0_10_fraction": float(np.mean(asr <= 0.10)),
        "median_ci_width": float(np.median(widths)),
        "mean_ci_width": float(np.mean(widths)),
        "zero_asr_95_upper_mean": float(np.mean(zero_high)) if len(zero_high) else None,
        "one_asr_95_lower_mean": float(np.mean(one_low)) if len(one_low) else None,
        "tx_histogram": {str(int(k)): int(v) for k, v in zip(*np.unique(tx_values, return_counts=True))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="统计经验 ASR 的低风险分辨率限制。")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.scores).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = row["dataset_key"]
        by_dataset[key].append(row)
        family = key.split("-", 1)[0]
        by_family[family].append(row)

    report = {
        "说明": "经验 ASR = tx/totQA。当前大多数 totQA=20，因此最小非零步长为 0.05；0.01 与 0.001 不能由单条指令的经验 ASR 直接区分。",
        "scores": args.scores,
        "all": summarize_rows(rows),
        "by_family": {key: summarize_rows(value) for key, value in sorted(by_family.items())},
        "by_dataset": {key: summarize_rows(value) for key, value in sorted(by_dataset.items())},
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "asr_resolution_limits.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 经验 ASR 分辨率限制", ""]
    lines.append("当前 txfilled 数据通常每条指令 `totQA=20`，因此经验 ASR 的最小非零步长是 `0.05`。")
    lines.append("这意味着 `0.01` 与 `0.001` 不能由单条指令的 20 次经验 ASR 直接区分，只能作为模型预测风险或序贯增采目标。")
    lines.append("")
    lines.extend(["## 总体", "", "| 指标 | 数值 |", "|---|---:|"])
    all_report = report["all"]
    for key in [
        "n",
        "unique_asr_count",
        "min_nonzero_step",
        "zero_asr_fraction",
        "low_asr_le_0_05_fraction",
        "low_asr_le_0_10_fraction",
        "median_ci_width",
        "zero_asr_95_upper_mean",
    ]:
        lines.append(f"| {key} | {all_report[key]} |")
    lines.append("")
    lines.extend(["## 按数据集族", "", "| 数据集族 | n | ASR均值 | 0档比例 | <=0.05比例 | 中位CI宽度 |", "|---|---:|---:|---:|---:|---:|"])
    for key, value in report["by_family"].items():
        lines.append(
            f"| {key} | {value['n']} | {value['asr_mean']:.3f} | {value['zero_asr_fraction']:.3f} | "
            f"{value['low_asr_le_0_05_fraction']:.3f} | {value['median_ci_width']:.3f} |"
        )
    lines.append("")
    lines.append("解释：当 `tx=0, totQA=20` 时，Jeffreys 95% 后验上界平均约为总体表中的 `zero_asr_95_upper_mean`，仍远大于 0.01。")
    md_path = out_dir / "经验ASR分辨率限制.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入 {json_path}")
    print(f"已写入 {md_path}")


if __name__ == "__main__":
    main()
