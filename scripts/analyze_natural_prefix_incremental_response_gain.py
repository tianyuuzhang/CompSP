#!/usr/bin/env python3
"""自然前缀下的回答文本条件增量实验。

本脚本落实可信度审计后的优先级：先随机排列回答并取自然出现的前 k 条，
不从完整 20 条回答中事后挑选 safe/unsafe 组成；随后比较
``count + q1`` 与 ``count + q1 + response`` 对未观察后缀 ASR/ALR 的预测能力。

核心问题不是“回答文本单独能不能预测”，而是：在已经知道前缀 unsafe/long 计数、
长度/拒绝/格式特征和原始攻击指令 q1 后，回答文本是否仍提供条件增量。

``toxic`` 与回答长度只用于构造自然前缀组成和未观察后缀标签，不进入文本特征。
输出包含问题级 cluster bootstrap 的 delta 置信区间，以及同组打乱 response 的负对照。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

from compsp.asr_dataset import iter_records
from compsp.response_structure import aggregate_answers, pairwise_accuracy_by_group, pairwise_accuracy_by_group_macro
from compsp.whitebox.supervised_direction import TEST_QIDS, TRAIN_QIDS, parse_csv_ints
from simulate_text_residual_sequential_asr import clean_text, select_base_features


def answer_is_unsafe(answer: dict) -> bool:
    """回答级 unsafe 判定，仅用于离线标签。"""

    return "unsafe" in str(answer.get("toxic", "")).lower()


def answer_len(answer: dict) -> int:
    """读取回答长度，缺失时退化为正文字符数。"""

    value = answer.get("len")
    if isinstance(value, (int, float)):
        return int(value)
    return len(str(answer.get("A", "")))


def rankdata(values: np.ndarray) -> np.ndarray:
    """平均秩实现，避免额外依赖。"""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 Spearman 相关。"""

    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(y_true), rankdata(y_pred))[0, 1])


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 Pearson 相关。"""

    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def build_rows(
    dataset_keys: list[str],
    question_ids: list[int],
    split_by_qid: dict[int, str],
    k: int,
    seed: int,
    length_threshold: int,
) -> list[dict]:
    """按自然随机前缀构造样本，后缀作为独立评估标签。"""

    rows: list[dict] = []
    for dataset_key in dataset_keys:
        for qid in question_ids:
            for item_index, record in enumerate(iter_records(dataset_key, [qid])):
                answers = [answer for answer in record.answers if isinstance(answer, dict)]
                if len(answers) <= k:
                    continue
                local_seed = seed * 1_000_003 + qid * 1009 + item_index * 917 + sum(map(ord, dataset_key))
                order = np.random.default_rng(local_seed).permutation(len(answers))
                prefix = [answers[int(index)] for index in order[:k]]
                suffix = [answers[int(index)] for index in order[k:]]
                prefix_tx = sum(answer_is_unsafe(answer) for answer in prefix)
                prefix_txl = sum(answer_len(answer) >= length_threshold for answer in prefix)
                future_tx = sum(answer_is_unsafe(answer) for answer in suffix)
                future_txl = sum(answer_len(answer) >= length_threshold for answer in suffix)
                features, response_text, extras = aggregate_answers(
                    prefix,
                    sample_size=k,
                    rng=np.random.default_rng(local_seed + 17),
                    long_threshold=length_threshold,
                )
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "question_id": qid,
                        "item_index": item_index,
                        "split": split_by_qid.get(qid, "test"),
                        "q1": record.q1,
                        "response_text": response_text,
                        "future_asr": future_tx / len(suffix),
                        "future_alr": future_txl / len(suffix),
                        "future_tx": int(future_tx),
                        "future_txl": int(future_txl),
                        "suffix_n": int(len(suffix)),
                        "prefix_tx": int(prefix_tx),
                        "prefix_txl": int(prefix_txl),
                        "prefix_asr_small": prefix_tx / len(prefix),
                        "prefix_alr_small": prefix_txl / len(prefix),
                        "composition": f"tx{prefix_tx}_txl{prefix_txl}",
                        "features": select_base_features(features),
                        **extras,
                    }
                )
    return rows


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


def fit_text_vectorizer(train_text: list[str], test_text: list[str], max_features: int, cleaning: str):
    """拟合 TF-IDF 文本特征。"""

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=max_features, sublinear_tf=True)
    x_train = vectorizer.fit_transform([clean_text(text, cleaning) for text in train_text])
    x_test = vectorizer.transform([clean_text(text, cleaning) for text in test_text])
    return x_train, x_test


def build_designs(train: list[dict], test: list[dict], max_features: int, cleaning: str) -> dict[str, tuple[csr_matrix, csr_matrix]]:
    """构造 count、count+q1、count+q1+response 三类设计矩阵。"""

    scaler = StandardScaler().fit(dense_matrix(train))
    dense_train = csr_matrix(scaler.transform(dense_matrix(train)))
    dense_test = csr_matrix(scaler.transform(dense_matrix(test)))
    q1_train, q1_test = fit_text_vectorizer(
        [row["q1"] for row in train],
        [row["q1"] for row in test],
        max_features=max(100, max_features // 2),
        cleaning="none",
    )
    response_train, response_test = fit_text_vectorizer(
        [row["response_text"] for row in train],
        [row["response_text"] for row in test],
        max_features=max(100, max_features // 2),
        cleaning=cleaning,
    )
    return {
        "count": (dense_train, dense_test),
        "count_q1": (hstack([dense_train, q1_train]).tocsr(), hstack([dense_test, q1_test]).tocsr()),
        "count_q1_response": (
            hstack([dense_train, q1_train, response_train]).tocsr(),
            hstack([dense_test, q1_test, response_test]).tocsr(),
        ),
    }


def shuffle_response_within_condition(rows: list[dict], seed: int) -> list[dict]:
    """在同数据集、同问题、同自然组成桶内打乱 response。"""

    rng = np.random.default_rng(seed)
    output = [dict(row) for row in rows]
    groups: dict[tuple[str, int, str], list[int]] = {}
    for index, row in enumerate(output):
        groups.setdefault((row["dataset_key"], row["question_id"], row["composition"]), []).append(index)
    for indices in groups.values():
        texts = [output[index]["response_text"] for index in indices]
        if len(texts) <= 1:
            continue
        permuted = [texts[int(index)] for index in rng.permutation(len(texts))]
        for index, text in zip(indices, permuted):
            output[index]["response_text"] = text
    return output


def evaluate(rows: list[dict], predictions: np.ndarray, target_key: str, high_threshold: float) -> dict[str, float]:
    """汇总回归、排序、高风险识别和问题宏平均指标。"""

    y = np.asarray([row[target_key] for row in rows], dtype=float)
    groups = [(row["dataset_key"], row["question_id"]) for row in rows]
    labels = (y >= high_threshold).astype(int)
    result = {
        "n": float(len(rows)),
        "question_n": float(len({group for group in groups})),
        "target_mean": float(np.mean(y)) if len(y) else float("nan"),
        "mae": float(mean_absolute_error(y, predictions)) if len(y) else float("nan"),
        "pearson": pearson(y, predictions),
        "spearman": spearman(y, predictions),
        "pairwise_acc_delta_0.1": pairwise_accuracy_by_group(y, predictions, groups, 0.1),
        "pairwise_macro_delta_0.1": pairwise_accuracy_by_group_macro(y, predictions, groups, 0.1),
        "high_rate": float(labels.mean()) if len(labels) else float("nan"),
    }
    if labels.sum() == 0 or labels.sum() == len(labels):
        result["ap"] = float("nan")
        result["auc"] = float("nan")
    else:
        result["ap"] = float(average_precision_score(labels, predictions))
        result["auc"] = float(roc_auc_score(labels, predictions))
    return result


def subgroup_rows(rows: list[dict], key: str) -> dict[str, list[int]]:
    """按指定字段生成子集下标。"""

    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(str(row[key]), []).append(index)
    return groups


def bootstrap_delta(
    rows: list[dict],
    base_pred: np.ndarray,
    full_pred: np.ndarray,
    target_key: str,
    metric: str,
    high_threshold: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    """按问题 cluster bootstrap 估计 full-base 的 delta 区间。"""

    if n_bootstrap <= 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    cluster_to_indices: dict[tuple[str, int], list[int]] = {}
    for index, row in enumerate(rows):
        cluster_to_indices.setdefault((row["dataset_key"], row["question_id"]), []).append(index)
    clusters = list(cluster_to_indices)
    if len(clusters) < 2:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_bootstrap):
        sampled = [clusters[int(index)] for index in rng.integers(0, len(clusters), size=len(clusters))]
        indices = [idx for cluster in sampled for idx in cluster_to_indices[cluster]]
        sample_rows = [rows[idx] for idx in indices]
        base_metrics = evaluate(sample_rows, base_pred[indices], target_key, high_threshold)
        full_metrics = evaluate(sample_rows, full_pred[indices], target_key, high_threshold)
        delta = full_metrics.get(metric, float("nan")) - base_metrics.get(metric, float("nan"))
        if np.isfinite(delta):
            deltas.append(delta)
    arr = np.asarray(deltas, dtype=float)
    return {
        "mean": float(np.mean(arr)) if len(arr) else float("nan"),
        "ci_low": float(np.quantile(arr, 0.025)) if len(arr) else float("nan"),
        "ci_high": float(np.quantile(arr, 0.975)) if len(arr) else float("nan"),
    }


def fit_predict(train: list[dict], test: list[dict], target_key: str, designs: dict[str, tuple[csr_matrix, csr_matrix]]) -> dict[str, np.ndarray]:
    """训练各模型并返回测试集预测。"""

    y_train = np.asarray([row[target_key] for row in train], dtype=float)
    predictions = {}
    for name, (x_train, x_test) in designs.items():
        predictions[name] = Ridge(alpha=10.0, solver="lsqr").fit(x_train, y_train).predict(x_test)
    return predictions


def aggregate_seed_metrics(values: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """对多 seed 指标取均值和样本标准差。"""

    keys = sorted({key for item in values for key in item})
    result: dict[str, dict[str, float]] = {}
    for key in keys:
        arr = np.asarray([item[key] for item in values if key in item and np.isfinite(item[key])], dtype=float)
        result[key] = {
            "mean": float(np.mean(arr)) if len(arr) else float("nan"),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }
    return result


def run_once(args: argparse.Namespace, k: int, seed: int) -> dict:
    """运行一个 k 与 seed。"""

    train_qids = args.resolved_train_qids
    test_qids = args.resolved_test_qids
    question_ids = sorted(set(train_qids + test_qids))
    split_by_qid = {qid: "train" for qid in train_qids} | {qid: "test" for qid in test_qids}
    rows = build_rows(args.dataset_keys, question_ids, split_by_qid, k, seed, args.length_threshold)
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    shuffled_test = shuffle_response_within_condition(test, seed + 20260713)
    designs = build_designs(train, test, args.max_features, args.text_cleaning)
    shuffled_designs = build_designs(train, shuffled_test, args.max_features, args.text_cleaning)

    output = {
        "k": k,
        "seed": seed,
        "train_n": len(train),
        "test_n": len(test),
        "composition_counts_test": {
            key: len(indices)
            for key, indices in sorted(subgroup_rows(test, "composition").items())
        },
        "targets": {},
    }
    for target_key in args.targets:
        predictions = fit_predict(train, test, target_key, designs)
        shuffled_predictions = fit_predict(train, shuffled_test, target_key, shuffled_designs)
        predictions["count_q1_response_shuffled"] = shuffled_predictions["count_q1_response"]
        target_report: dict[str, dict] = {
            name: evaluate(test, pred, target_key, args.high_threshold)
            for name, pred in predictions.items()
        }
        target_report["delta_full_minus_count_q1"] = {
            metric: bootstrap_delta(
                test,
                predictions["count_q1"],
                predictions["count_q1_response"],
                target_key,
                metric,
                args.high_threshold,
                args.bootstrap,
                seed + k * 1000,
            )
            for metric in ["spearman", "ap", "pairwise_macro_delta_0.1"]
        }
        target_report["by_attack"] = {}
        for attack, indices in subgroup_rows(test, "dataset_key").items():
            if len(indices) < args.min_subgroup_n:
                continue
            sub_rows = [test[idx] for idx in indices]
            target_report["by_attack"][attack] = {
                name: evaluate(sub_rows, pred[indices], target_key, args.high_threshold)
                for name, pred in predictions.items()
            }
        target_report["by_composition"] = {}
        for composition, indices in subgroup_rows(test, "composition").items():
            if len(indices) < args.min_subgroup_n:
                continue
            sub_rows = [test[idx] for idx in indices]
            target_report["by_composition"][composition] = {
                name: evaluate(sub_rows, pred[indices], target_key, args.high_threshold)
                for name, pred in predictions.items()
            }
        output["targets"][target_key] = target_report
    return output


def summarize_seed_outputs(seed_outputs: list[dict], targets: list[str]) -> dict:
    """汇总多 seed 主指标。"""

    summary = {}
    for target in targets:
        model_names = [
            name
            for name in seed_outputs[0]["targets"][target]
            if name not in {"delta_full_minus_count_q1", "by_attack", "by_composition"}
        ]
        summary[target] = {
            name: aggregate_seed_metrics([item["targets"][target][name] for item in seed_outputs])
            for name in model_names
        }
        summary[target]["delta_full_minus_count_q1"] = {
            metric: aggregate_seed_metrics([
                item["targets"][target]["delta_full_minus_count_q1"][metric]
                for item in seed_outputs
            ])
            for metric in ["spearman", "ap", "pairwise_macro_delta_0.1"]
        }
    return summary


def write_markdown(report: dict, path: Path) -> None:
    """写中文 Markdown 摘要。"""

    lines = [
        "# 自然前缀回答文本条件增量实验",
        "",
        "本实验先随机排列回答并取自然前缀，不从完整回答池事后挑选 safe/unsafe 组成。",
        "主比较是 `count_q1` 与 `count_q1_response`，即在前缀计数、长度格式和 q1 后，回答文本是否仍有增量。",
        "",
    ]
    for k, k_report in report["sample_sizes"].items():
        lines.extend([f"## k={k}", ""])
        for target, target_report in k_report["summary"].items():
            lines.extend([
                f"### 目标：{target}",
                "",
                "| 模型 | n | 问题数 | MAE | Spearman | macro pairwise@0.1 | AP | AUC |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for name, metrics in target_report.items():
                if name == "delta_full_minus_count_q1":
                    continue
                lines.append(
                    f"| {name} | {metrics['n']['mean']:.0f} | {metrics['question_n']['mean']:.0f} | "
                    f"{metrics['mae']['mean']:.3f} | {metrics['spearman']['mean']:.3f} | "
                    f"{metrics['pairwise_macro_delta_0.1']['mean']:.3f} | "
                    f"{metrics['ap']['mean']:.3f} | {metrics['auc']['mean']:.3f} |"
                )
            lines.append("")
            lines.append("| delta 指标 | bootstrap delta mean | 95% CI |")
            lines.append("|---|---:|---:|")
            for metric, values in target_report["delta_full_minus_count_q1"].items():
                lines.append(
                    f"| {metric} | {values['mean']['mean']:.3f} | "
                    f"[{values['ci_low']['mean']:.3f}, {values['ci_high']['mean']:.3f}] |"
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="自然前缀下 count+q1+response 条件增量实验。")
    parser.add_argument("--dataset-keys", default="jbb-llama-ofa,jbb-llama-pair,jbb-llama-drattack")
    parser.add_argument("--train-question-ids", default=",".join(map(str, sorted(TRAIN_QIDS))))
    parser.add_argument("--test-question-ids", default=",".join(map(str, sorted(TEST_QIDS))))
    parser.add_argument("--split-mode", choices=["fixed", "random80"], default="fixed")
    parser.add_argument("--split-seed", type=int, default=20260713)
    parser.add_argument("--sample-sizes", default="1,2,4")
    parser.add_argument("--targets", default="future_asr")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--min-subgroup-n", type=int, default=80)
    parser.add_argument("--length-threshold", type=int, default=500)
    parser.add_argument("--high-threshold", type=float, default=0.5)
    parser.add_argument("--max-features", type=int, default=12000)
    parser.add_argument("--text-cleaning", default="mask_strong_artifacts")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    args.dataset_keys = [key.strip() for key in args.dataset_keys.split(",") if key.strip()]
    args.targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    if args.split_mode == "random80":
        all_qids = np.arange(80)
        rng = np.random.default_rng(args.split_seed)
        test_qids = sorted(rng.choice(all_qids, size=20, replace=False).astype(int).tolist())
        train_qids = sorted([int(qid) for qid in all_qids.tolist() if int(qid) not in set(test_qids)])
    else:
        train_qids = parse_csv_ints(args.train_question_ids, sorted(TRAIN_QIDS))
        test_qids = parse_csv_ints(args.test_question_ids, sorted(TEST_QIDS))
    args.resolved_train_qids = train_qids
    args.resolved_test_qids = test_qids
    sample_sizes = [int(value) for value in args.sample_sizes.split(",") if value.strip()]
    report = {
        "说明": "自然前缀下比较 count+q1 与 count+q1+response 的条件增量；标签只使用未观察后缀。",
        "datasets": args.dataset_keys,
        "split_mode": args.split_mode,
        "split_seed": args.split_seed,
        "train_questions": train_qids,
        "test_questions": test_qids,
        "targets": args.targets,
        "text_cleaning": args.text_cleaning,
        "bootstrap": args.bootstrap,
        "sample_sizes": {},
    }
    for k in sample_sizes:
        seed_outputs = []
        for seed in range(args.seed_offset, args.seed_offset + args.seeds):
            seed_outputs.append(run_once(args, k, seed))
            print(f"完成 k={k}, seed={seed}", flush=True)
        report["sample_sizes"][str(k)] = {
            "seeds": seed_outputs,
            "summary": summarize_seed_outputs(seed_outputs, args.targets),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, output.with_suffix(".md"))
    print(f"JSON 已写入 {output}")
    print(f"Markdown 已写入 {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
