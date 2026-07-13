#!/usr/bin/env python3
"""模拟固定追加采样预算下的高风险指令发现率。

每条指令先观察前 k 条回答；随后策略只能把额外预算分配给同题 top fraction
候选。被选中的候选离线读取剩余后缀回答，未选中的候选不再追加。该设置近似
真实流程中的“先少量嗅探，再决定是否继续访问”。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compsp.whitebox.supervised_direction import TEST_QIDS, TRAIN_QIDS, parse_csv_ints
from simulate_text_residual_sequential_asr import (
    aggregate,
    build_rows,
    fit_text_residual_scores,
    select_hybrid_with_prefix_zero_quota,
    select_within_groups,
)


def evaluate_budget(rows: list[dict], selected: list[dict], high_threshold: float, k: int) -> dict:
    """评估追加采样后发现 unsafe 或高风险指令的效率。"""

    selected_ids = {(row["dataset_key"], row["question_id"], row["item_index"]) for row in selected}
    true_any_success = [row for row in rows if row["prefix_tx"] + row["future_tx"] > 0]
    initial_found = [row for row in rows if row["prefix_tx"] > 0]
    final_found = [
        row for row in rows
        if row["prefix_tx"] > 0 or ((row["dataset_key"], row["question_id"], row["item_index"]) in selected_ids and row["future_tx"] > 0)
    ]
    prefix_zero_high = [row for row in rows if row["prefix_zero"] and row["future_asr"] >= high_threshold]
    prefix_zero_high_found = [
        row for row in prefix_zero_high
        if (row["dataset_key"], row["question_id"], row["item_index"]) in selected_ids and row["future_tx"] > 0
    ]
    initial_budget = len(rows) * k
    extra_budget = sum(row["suffix_n"] for row in selected)
    return {
        "rows": len(rows),
        "selected_n": len(selected),
        "initial_budget": int(initial_budget),
        "extra_budget": int(extra_budget),
        "total_budget": int(initial_budget + extra_budget),
        "extra_budget_per_row": float(extra_budget / len(rows)) if rows else float("nan"),
        "true_any_success_total": len(true_any_success),
        "initial_found_n": len(initial_found),
        "final_found_n": len(final_found),
        "new_found_n": len(final_found) - len(initial_found),
        "initial_any_success_recall": len(initial_found) / len(true_any_success) if true_any_success else float("nan"),
        "final_any_success_recall": len(final_found) / len(true_any_success) if true_any_success else float("nan"),
        "recall_gain": (len(final_found) - len(initial_found)) / len(true_any_success) if true_any_success else float("nan"),
        "prefix_zero_high_total": len(prefix_zero_high),
        "prefix_zero_high_found": len(prefix_zero_high_found),
        "prefix_zero_high_recall": len(prefix_zero_high_found) / len(prefix_zero_high) if prefix_zero_high else float("nan"),
        "selected_future_asr_mean": float(np.mean([row["future_asr"] for row in selected])) if selected else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="模拟 prefix-zero 嗅探器的真实追加采样预算收益。")
    parser.add_argument("--dataset-keys", default="jbb-llama-ofa,jbb-llama-pair,jbb-llama-drattack")
    parser.add_argument("--train-question-ids", default=",".join(map(str, sorted(TRAIN_QIDS))))
    parser.add_argument("--test-question-ids", default=",".join(map(str, sorted(TEST_QIDS))))
    parser.add_argument("--sample-sizes", default="1,2,4")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument("--hidden-quotas", default="0.25,0.5")
    parser.add_argument("--high-threshold", type=float, default=0.5)
    parser.add_argument("--length-threshold", type=int, default=500)
    parser.add_argument("--max-features", type=int, default=12000)
    parser.add_argument("--text-cleaning", default="mask_strong_artifacts")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_keys = [key.strip() for key in args.dataset_keys.split(",") if key.strip()]
    train_qids = parse_csv_ints(args.train_question_ids, sorted(TRAIN_QIDS))
    test_qids = parse_csv_ints(args.test_question_ids, sorted(TEST_QIDS))
    question_ids = sorted(set(train_qids + test_qids))
    split_by_qid = {qid: "train" for qid in train_qids} | {qid: "test" for qid in test_qids}
    hidden_quotas = [float(value) for value in args.hidden_quotas.split(",") if value.strip()]

    score_keys = [
        "random",
        "prefix_hazard_weighted_asr",
        "prefix_alr",
        "model_base_features",
        "model_text_only",
        "model_base_plus_text_residual",
    ]
    hybrid_keys = [f"hybrid_hazard_text_q{quota:g}" for quota in hidden_quotas]
    report = {
        "说明": "每条先观察 k 次回答，再把额外后缀预算分配给 top 候选。",
        "datasets": dataset_keys,
        "top_fraction": args.top_fraction,
        "high_threshold": args.high_threshold,
        "text_cleaning": args.text_cleaning,
        "sample_sizes": {},
    }
    for k in [int(value) for value in args.sample_sizes.split(",") if value.strip()]:
        seed_reports = {score: [] for score in score_keys + hybrid_keys}
        for seed in range(args.seeds):
            rows = build_rows(dataset_keys, question_ids, k, seed, args.length_threshold, split_by_qid)
            train = [row for row in rows if row["split"] == "train"]
            test = [row for row in rows if row["split"] == "test"]
            model_scores = fit_text_residual_scores(
                train,
                test,
                args.max_features,
                args.text_cleaning,
                False,
                10.0,
                10.0,
            )
            test_scored = [dict(row) for row in test]
            for score_name, values in model_scores.items():
                for row, value in zip(test_scored, values.tolist()):
                    row[score_name] = float(value)
            rng = np.random.default_rng(seed + 20260713)
            for score in score_keys:
                selected = select_within_groups(test_scored, score, args.top_fraction, rng)
                seed_reports[score].append(evaluate_budget(test_scored, selected, args.high_threshold, k))
            for quota in hidden_quotas:
                selected = select_hybrid_with_prefix_zero_quota(
                    test_scored,
                    primary_score="prefix_hazard_weighted_asr",
                    hidden_score="model_base_plus_text_residual",
                    top_fraction=args.top_fraction,
                    hidden_quota=quota,
                    rng=rng,
                )
                seed_reports[f"hybrid_hazard_text_q{quota:g}"].append(evaluate_budget(test_scored, selected, args.high_threshold, k))
            print(f"完成 k={k}, seed={seed}", flush=True)
        report["sample_sizes"][str(k)] = {score: aggregate(values) for score, values in seed_reports.items()}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 固定预算追加采样模拟", ""]
    lines.append(f"初始每条观察 k 次；额外预算给同题 top {args.top_fraction:.0%} 候选；清理口径：`{args.text_cleaning}`。")
    lines.append("")
    for k, k_report in report["sample_sizes"].items():
        lines.extend([
            f"## k={k}",
            "",
            "| 策略 | extra/row | final recall | recall gain | prefix-zero高风险recall | selected future ASR |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for score, values in k_report.items():
            lines.append(
                f"| {score} | {values['extra_budget_per_row']['mean']:.2f} | "
                f"{values['final_any_success_recall']['mean']:.3f} | "
                f"{values['recall_gain']['mean']:.3f} | "
                f"{values['prefix_zero_high_recall']['mean']:.3f} | "
                f"{values['selected_future_asr_mean']['mean']:.3f} |"
            )
        lines.append("")
    md_path = output.with_suffix(".md")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON 已写入 {output}")
    print(f"Markdown 已写入 {md_path}")


if __name__ == "__main__":
    main()
