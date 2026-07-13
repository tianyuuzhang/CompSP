#!/usr/bin/env python3
"""固定追加访问预算下的高风险指令发现曲线。

旧版序贯模拟按同题 top 20% 选择候选。该口径适合回答“选前多少比例是否有效”，
但不同 k 的后缀长度不同，且固定比例不等价于固定访问成本。本脚本改为给每个策略
完全相同的 extra response 预算：每条指令先观察前 k 条回答，再在测试问题内按策略
排序追加访问，直到用完指定的平均追加预算。

输入特征仍只来自攻击者可见的前缀回答文本和文本统计特征；unsafe/toxic 字段只用于
离线评价。原始数据只读，输出写入 outputs。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compsp.whitebox.supervised_direction import TEST_QIDS, TRAIN_QIDS, parse_csv_ints
from simulate_budgeted_asr_discovery import evaluate_budget
from simulate_text_residual_sequential_asr import (
    aggregate,
    build_rows,
    fit_text_residual_scores,
)


def group_rows(rows: list[dict]) -> dict[tuple[str, int], list[dict]]:
    """按数据集和问题分组，避免跨题或跨攻击方案互相竞争预算。"""

    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["dataset_key"], row["question_id"]), []).append(row)
    return groups


def take_until_budget(ranked: list[dict], budget: int) -> list[dict]:
    """按排序依次选择候选，保证实际消耗不超过预算。"""

    selected: list[dict] = []
    used = 0
    for row in ranked:
        cost = int(row["suffix_n"])
        if cost <= 0:
            continue
        if used + cost > budget:
            continue
        selected.append(row)
        used += cost
    return selected


def select_group_budget(
    rows: list[dict],
    score_key: str,
    extra_budget_per_row: float,
    rng: np.random.Generator,
) -> list[dict]:
    """在每个数据集-问题组内使用固定平均追加预算选择候选。"""

    selected: list[dict] = []
    for group in group_rows(rows).values():
        budget = int(round(len(group) * extra_budget_per_row))
        if budget <= 0:
            continue
        if score_key == "random":
            ranked = [group[int(index)] for index in rng.permutation(len(group))]
        else:
            ranked = sorted(group, key=lambda item: (item[score_key], rng.random()), reverse=True)
        selected.extend(take_until_budget(ranked, budget))
    return selected


def select_hybrid_group_budget(
    rows: list[dict],
    primary_score: str,
    hidden_score: str,
    extra_budget_per_row: float,
    hidden_quota: float,
    rng: np.random.Generator,
) -> list[dict]:
    """固定预算混合策略：为 prefix-zero 样本保留一部分追加预算。"""

    selected: list[dict] = []
    for group in group_rows(rows).values():
        total_budget = int(round(len(group) * extra_budget_per_row))
        hidden_budget = int(round(total_budget * hidden_quota))
        if total_budget <= 0:
            continue

        prefix_zero = [row for row in group if row["prefix_zero"]]
        hidden_ranked = sorted(prefix_zero, key=lambda item: (item[hidden_score], rng.random()), reverse=True)
        group_selected = take_until_budget(hidden_ranked, hidden_budget)
        selected_ids = {
            (row["dataset_key"], row["question_id"], row["item_index"])
            for row in group_selected
        }
        used = sum(int(row["suffix_n"]) for row in group_selected)
        remaining_budget = max(0, total_budget - used)
        remaining = [
            row for row in group
            if (row["dataset_key"], row["question_id"], row["item_index"]) not in selected_ids
        ]
        primary_ranked = sorted(remaining, key=lambda item: (item[primary_score], rng.random()), reverse=True)
        group_selected.extend(take_until_budget(primary_ranked, remaining_budget))
        selected.extend(group_selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="固定 extra response 预算的序贯 ASR 发现曲线。")
    parser.add_argument("--dataset-keys", default="jbb-llama-ofa,jbb-llama-pair,jbb-llama-drattack")
    parser.add_argument("--train-question-ids", default=",".join(map(str, sorted(TRAIN_QIDS))))
    parser.add_argument("--test-question-ids", default=",".join(map(str, sorted(TEST_QIDS))))
    parser.add_argument("--sample-sizes", default="1,2,4")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--extra-budget-per-row", default="1,2,3,4")
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
    budgets = [float(value) for value in args.extra_budget_per_row.split(",") if value.strip()]
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
        "说明": "每条先观察 k 次回答；每个数据集-问题组获得相同平均 extra response 预算，然后按策略排序追加访问。",
        "datasets": dataset_keys,
        "train_questions": train_qids,
        "test_questions": test_qids,
        "extra_budget_per_row_grid": budgets,
        "hidden_quotas": hidden_quotas,
        "high_threshold": args.high_threshold,
        "text_cleaning": args.text_cleaning,
        "sample_sizes": {},
    }

    for k in [int(value) for value in args.sample_sizes.split(",") if value.strip()]:
        budget_reports = {
            str(budget): {score: [] for score in score_keys + hybrid_keys}
            for budget in budgets
        }
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
            for budget in budgets:
                rng = np.random.default_rng(seed + 20260713 + int(round(budget * 1000)))
                for score in score_keys:
                    selected = select_group_budget(test_scored, score, budget, rng)
                    budget_reports[str(budget)][score].append(evaluate_budget(test_scored, selected, args.high_threshold, k))
                for quota in hidden_quotas:
                    selected = select_hybrid_group_budget(
                        test_scored,
                        primary_score="prefix_hazard_weighted_asr",
                        hidden_score="model_base_plus_text_residual",
                        extra_budget_per_row=budget,
                        hidden_quota=quota,
                        rng=rng,
                    )
                    budget_reports[str(budget)][f"hybrid_hazard_text_q{quota:g}"].append(
                        evaluate_budget(test_scored, selected, args.high_threshold, k)
                    )
                print(f"完成 k={k}, extra/row={budget:g}, seed={seed}", flush=True)
        report["sample_sizes"][str(k)] = {
            str(budget): {
                score: aggregate(values)
                for score, values in budget_reports[str(budget)].items()
            }
            for budget in budgets
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 固定 extra response 预算的 ASR 发现曲线", ""]
    lines.append(
        f"每条先观察前缀 `k` 次；每个数据集-问题组按 `extra_budget_per_row` 获得相同追加访问预算；"
        f"清理口径：`{args.text_cleaning}`。"
    )
    lines.append("")
    for k, k_report in report["sample_sizes"].items():
        for budget, budget_report in k_report.items():
            lines.extend([
                f"## k={k}，目标 extra/row={float(budget):g}",
                "",
                "| 策略 | 实际extra/row | final recall | recall gain | prefix-zero高风险recall | selected future ASR | selected_n |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ])
            for score, values in budget_report.items():
                lines.append(
                    f"| {score} | {values['extra_budget_per_row']['mean']:.2f} | "
                    f"{values['final_any_success_recall']['mean']:.3f} | "
                    f"{values['recall_gain']['mean']:.3f} | "
                    f"{values['prefix_zero_high_recall']['mean']:.3f} | "
                    f"{values['selected_future_asr_mean']['mean']:.3f} | "
                    f"{values['selected_n']['mean']:.1f} |"
                )
            lines.append("")
    md_path = output.with_suffix(".md")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON 已写入 {output}")
    print(f"Markdown 已写入 {md_path}")


if __name__ == "__main__":
    main()
