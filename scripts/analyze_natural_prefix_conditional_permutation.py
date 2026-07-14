#!/usr/bin/env python3
"""自然前缀回答文本条件置换检验。

本脚本在 ``count + q1`` 条件下检验 response 的增量是否超过同条件置换噪声。
每个 k/seed 只训练一次嵌套模型，然后在测试集的同数据集、同问题、同自然组成桶内
多次打乱 response，重算 ``count + q1 + response`` 预测，形成 delta 的 null 分布。

这不是新采样确认，只是回答“当前自然前缀实验中的 response 增量是否显著强于
条件打乱 response”。原始数据只读，输出写入 outputs。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from analyze_natural_prefix_incremental_response_gain import (
    build_rows,
    clean_text,
    evaluate,
    subgroup_rows,
)
from compsp.whitebox.supervised_direction import TEST_QIDS, TRAIN_QIDS, parse_csv_ints


def resolve_split(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    """解析固定或 random80 问题划分。"""

    if args.split_mode == "random80":
        all_qids = np.arange(80)
        rng = np.random.default_rng(args.split_seed)
        test_qids = sorted(rng.choice(all_qids, size=20, replace=False).astype(int).tolist())
        train_qids = sorted([int(qid) for qid in all_qids.tolist() if int(qid) not in set(test_qids)])
    else:
        train_qids = parse_csv_ints(args.train_question_ids, sorted(TRAIN_QIDS))
        test_qids = parse_csv_ints(args.test_question_ids, sorted(TEST_QIDS))
    return train_qids, test_qids


def dense_matrix(rows: list[dict]) -> np.ndarray:
    """构造 count、比例和长度格式基线特征。"""

    scalar = np.asarray(
        [
            [
                row["prefix_tx"],
                row["prefix_txl"],
                row["prefix_asr_small"],
                row["prefix_alr_small"],
                row["sample_long_ratio"],
                row["sample_refusal_ratio"],
            ]
            for row in rows
        ],
        dtype=float,
    )
    feature = np.stack([row["features"] for row in rows])
    return np.concatenate([scalar, feature], axis=1)


def fit_once(train: list[dict], test: list[dict], target_key: str, max_features: int, cleaning: str):
    """训练 base/full 模型，并返回复用对象。"""

    scaler = StandardScaler().fit(dense_matrix(train))
    dense_train = csr_matrix(scaler.transform(dense_matrix(train)))
    dense_test = csr_matrix(scaler.transform(dense_matrix(test)))
    q1_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=max(100, max_features // 2), sublinear_tf=True)
    q1_train = q1_vec.fit_transform([row["q1"] for row in train])
    q1_test = q1_vec.transform([row["q1"] for row in test])
    resp_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=max(100, max_features // 2), sublinear_tf=True)
    resp_train = resp_vec.fit_transform([clean_text(row["response_text"], cleaning) for row in train])
    resp_test = resp_vec.transform([clean_text(row["response_text"], cleaning) for row in test])
    y_train = np.asarray([row[target_key] for row in train], dtype=float)
    x_base_train = hstack([dense_train, q1_train]).tocsr()
    x_base_test = hstack([dense_test, q1_test]).tocsr()
    x_full_train = hstack([dense_train, q1_train, resp_train]).tocsr()
    x_full_test = hstack([dense_test, q1_test, resp_test]).tocsr()
    base_model = Ridge(alpha=10.0, solver="lsqr").fit(x_base_train, y_train)
    full_model = Ridge(alpha=10.0, solver="lsqr").fit(x_full_train, y_train)
    return {
        "dense_test": dense_test,
        "q1_test": q1_test,
        "resp_vec": resp_vec,
        "base_pred": base_model.predict(x_base_test),
        "full_pred": full_model.predict(x_full_test),
        "full_model": full_model,
    }


def permuted_response_texts(rows: list[dict], seed: int) -> list[str]:
    """在同数据集、同问题、同自然组成桶内置换 response 文本。"""

    rng = np.random.default_rng(seed)
    texts = [row["response_text"] for row in rows]
    output = list(texts)
    groups: dict[tuple[str, int, str], list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault((row["dataset_key"], row["question_id"], row["composition"]), []).append(index)
    for indices in groups.values():
        if len(indices) <= 1:
            continue
        local = [texts[index] for index in indices]
        permuted = [local[int(index)] for index in rng.permutation(len(local))]
        for index, text in zip(indices, permuted):
            output[index] = text
    return output


def metric_delta(rows: list[dict], base_pred: np.ndarray, full_pred: np.ndarray, target_key: str, high_threshold: float) -> dict[str, float]:
    """计算 full-base 的主要 delta。"""

    base = evaluate(rows, base_pred, target_key, high_threshold)
    full = evaluate(rows, full_pred, target_key, high_threshold)
    return {
        "spearman": full["spearman"] - base["spearman"],
        "ap": full["ap"] - base["ap"],
        "pairwise_macro_delta_0.1": full["pairwise_macro_delta_0.1"] - base["pairwise_macro_delta_0.1"],
    }


def summarize(values: list[float]) -> dict[str, float]:
    """汇总均值和区间。"""

    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "q025": float(np.quantile(arr, 0.025)),
        "q975": float(np.quantile(arr, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="自然前缀 response 条件置换检验。")
    parser.add_argument("--dataset-keys", default="jbb-llama-ofa,jbb-llama-pair,jbb-llama-drattack")
    parser.add_argument("--train-question-ids", default=",".join(map(str, sorted(TRAIN_QIDS))))
    parser.add_argument("--test-question-ids", default=",".join(map(str, sorted(TEST_QIDS))))
    parser.add_argument("--split-mode", choices=["fixed", "random80"], default="fixed")
    parser.add_argument("--split-seed", type=int, default=20260713)
    parser.add_argument("--sample-sizes", default="1,2,4")
    parser.add_argument("--target", default="future_asr")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--length-threshold", type=int, default=500)
    parser.add_argument("--high-threshold", type=float, default=0.5)
    parser.add_argument("--max-features", type=int, default=6000)
    parser.add_argument("--text-cleaning", default="mask_strong_artifacts")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_keys = [key.strip() for key in args.dataset_keys.split(",") if key.strip()]
    train_qids, test_qids = resolve_split(args)
    split_by_qid = {qid: "train" for qid in train_qids} | {qid: "test" for qid in test_qids}
    question_ids = sorted(set(train_qids + test_qids))
    report = {
        "说明": "在 count+q1 条件下，同数据集/同问题/同自然组成桶置换 response，检验 response 增量。",
        "datasets": dataset_keys,
        "split_mode": args.split_mode,
        "split_seed": args.split_seed,
        "train_questions": train_qids,
        "test_questions": test_qids,
        "target": args.target,
        "permutations": args.permutations,
        "sample_sizes": {},
    }
    for k in [int(value) for value in args.sample_sizes.split(",") if value.strip()]:
        observed_by_metric = {"spearman": [], "ap": [], "pairwise_macro_delta_0.1": []}
        null_by_metric = {"spearman": [], "ap": [], "pairwise_macro_delta_0.1": []}
        seed_reports = []
        for seed in range(args.seeds):
            rows = build_rows(dataset_keys, question_ids, split_by_qid, k, seed, args.length_threshold)
            train = [row for row in rows if row["split"] == "train"]
            test = [row for row in rows if row["split"] == "test"]
            fit = fit_once(train, test, args.target, args.max_features, args.text_cleaning)
            observed = metric_delta(test, fit["base_pred"], fit["full_pred"], args.target, args.high_threshold)
            for metric, value in observed.items():
                observed_by_metric[metric].append(value)
            permuted_metrics = []
            for perm_id in range(args.permutations):
                texts = permuted_response_texts(test, seed * 100000 + k * 1000 + perm_id)
                resp_test = fit["resp_vec"].transform([clean_text(text, args.text_cleaning) for text in texts])
                x_perm = hstack([fit["dense_test"], fit["q1_test"], resp_test]).tocsr()
                pred = fit["full_model"].predict(x_perm)
                delta = metric_delta(test, fit["base_pred"], pred, args.target, args.high_threshold)
                permuted_metrics.append(delta)
                for metric, value in delta.items():
                    null_by_metric[metric].append(value)
            seed_reports.append(
                {
                    "seed": seed,
                    "train_n": len(train),
                    "test_n": len(test),
                    "composition_counts_test": {
                        key: len(indices)
                        for key, indices in sorted(subgroup_rows(test, "composition").items())
                    },
                    "observed_delta": observed,
                    "null_delta": {
                        metric: summarize([item[metric] for item in permuted_metrics])
                        for metric in observed
                    },
                }
            )
            print(f"完成 k={k}, seed={seed}", flush=True)
        report["sample_sizes"][str(k)] = {
            "seeds": seed_reports,
            "observed_delta": {metric: summarize(values) for metric, values in observed_by_metric.items()},
            "null_delta": {metric: summarize(values) for metric, values in null_by_metric.items()},
            "p_value_greater": {
                metric: float((1 + np.sum(np.asarray(null_by_metric[metric]) >= np.mean(observed_by_metric[metric]))) / (1 + len(null_by_metric[metric])))
                for metric in observed_by_metric
            },
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 自然前缀 response 条件置换检验",
        "",
        f"split_mode=`{args.split_mode}`，split_seed=`{args.split_seed}`，permutations/seed=`{args.permutations}`。",
        "",
        "| k | 指标 | observed mean | null mean | null 95% | p(null>=obs) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for k, item in report["sample_sizes"].items():
        for metric in ["spearman", "ap", "pairwise_macro_delta_0.1"]:
            obs = item["observed_delta"][metric]
            null = item["null_delta"][metric]
            lines.append(
                f"| {k} | {metric} | {obs['mean']:.4f} | {null['mean']:.4f} | "
                f"[{null['q025']:.4f}, {null['q975']:.4f}] | {item['p_value_greater'][metric]:.4f} |"
            )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON 已写入 {output}")
    print(f"Markdown 已写入 {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
