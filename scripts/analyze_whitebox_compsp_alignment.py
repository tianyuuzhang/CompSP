#!/usr/bin/env python3
"""分析白盒分数、CompSP BTL 与 ASR/ALR 的对齐关系。"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from compsp.whitebox.ranking_io import load_btl_scores


def load_rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_spearman(a, b) -> float | None:
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    val = spearmanr(a, b).correlation
    return None if np.isnan(val) else float(val)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--project-root", default="/remote-home/zty/tidy_CompSP")
    parser.add_argument("--metric", choices=["asr", "alr"], default="alr")
    parser.add_argument("--min-delta", type=float, default=0.15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = load_rows(args.scores)
    by_q: dict[int, list[dict]] = {}
    for row in rows:
        by_q.setdefault(int(row["question_id"]), []).append(row)

    merged = []
    for qid, qrows in by_q.items():
        btl = load_btl_scores(args.project_root, args.dataset_key, qid)
        for row in qrows:
            item_index = int(row["item_index"])
            if item_index in btl:
                row["compsp_btl"] = btl[item_index]
                merged.append(row)

    if not merged:
        raise RuntimeError("没有任何行匹配到 CompSP BTL 分数")

    n_layers = len(merged[0]["wb_last"])
    layer_stats = []
    for layer in range(n_layers):
        wb = [r["wb_last"][layer] for r in merged]
        metric = [r[args.metric] for r in merged]
        btl = [r["compsp_btl"] for r in merged]
        layer_stats.append(
            {
                "layer": layer,
                "spearman_wb_metric": safe_spearman(wb, metric),
                "spearman_btl_metric": safe_spearman(btl, metric),
                "spearman_wb_btl": safe_spearman(wb, btl),
            }
        )

    pair_stats = []
    for layer in range(n_layers):
        for sign in (1, -1):
            total = compsp_correct = wb_correct = wb_compsp_agree = compsp_wrong = wrong_agree = 0
            for qid, qrows in by_q.items():
                qrows = [r for r in qrows if "compsp_btl" in r]
                for a, b in itertools.combinations(qrows, 2):
                    delta = a[args.metric] - b[args.metric]
                    if abs(delta) <= args.min_delta:
                        continue
                    empirical = delta > 0
                    compsp = a["compsp_btl"] > b["compsp_btl"]
                    wb = sign * a["wb_last"][layer] > sign * b["wb_last"][layer]
                    total += 1
                    compsp_correct += int(compsp == empirical)
                    wb_correct += int(wb == empirical)
                    wb_compsp_agree += int(wb == compsp)
                    if compsp != empirical:
                        compsp_wrong += 1
                        wrong_agree += int(wb == compsp)
            pair_stats.append(
                {
                    "layer": layer,
                    "sign": sign,
                    "pairs": total,
                    "compsp_acc": compsp_correct / total if total else None,
                    "wb_acc": wb_correct / total if total else None,
                    "wb_compsp_agreement": wb_compsp_agree / total if total else None,
                    "compsp_wrong_pairs": compsp_wrong,
                    "wrong_agreement": wrong_agree / compsp_wrong if compsp_wrong else None,
                }
            )

    report = {
        "dataset_key": args.dataset_key,
        "metric": args.metric,
        "rows_scored": len(rows),
        "rows_matched_btl": len(merged),
        "layer_stats": layer_stats,
        "pair_stats": pair_stats,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    best = max(pair_stats, key=lambda x: -1 if x["wrong_agreement"] is None else x["wrong_agreement"])
    print(f"已写入 {out}; matched={len(merged)}; best_wrong_agreement={best}")


if __name__ == "__main__":
    main()
