#!/usr/bin/env python3
"""分析回答内容、ASR/ALR 与白盒伪安全投影的三方关联。

脚本使用固定 60/20 问题划分。特征仅来自回答正文；安全判定字段不会进入模型。
默认先运行可解释的手工特征 Ridge，指定 ``--methods tfidf`` 后再加入去拒绝前缀
文本的词/字符 n-gram 模型。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from compsp.asr_dataset import iter_records
from compsp.response_structure import FEATURE_NAMES, aggregate_answers, pairwise_accuracy_by_group
from compsp.whitebox.supervised_direction import read_jsonl


def safe_corr(left: np.ndarray, right: np.ndarray, method: str) -> float:
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    result = pearsonr(left, right) if method == "pearson" else spearmanr(left, right)
    return float(result.statistic)


def metrics(y: np.ndarray, pred: np.ndarray, groups: list[tuple[str, int]]) -> dict[str, float]:
    return {
        "n": int(len(y)),
        "pearson": safe_corr(pred, y, "pearson"),
        "spearman": safe_corr(pred, y, "spearman"),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "同题序关系准确率": pairwise_accuracy_by_group(y, pred, groups),
    }


def sliced_metrics(
    test: list[dict], y: np.ndarray, pred: np.ndarray, groups: list[tuple[str, int]]
) -> dict[str, dict[str, float]]:
    """同时报告三种攻击方案的分项结果，防止模板差异形成伪相关。"""

    result = {"合并": metrics(y, pred, groups)}
    for dataset_key in sorted({row["dataset_key"] for row in test}):
        mask = np.asarray([row["dataset_key"] == dataset_key for row in test], dtype=bool)
        result[dataset_key] = metrics(
            y[mask], pred[mask], [group for group, keep in zip(groups, mask.tolist()) if keep]
        )
    return result


def load_rows(
    score_file: str,
    sample_size: int,
    seed: int,
    max_rows: int | None,
    target_names: list[str],
    dataset_filter: set[str] | None = None,
) -> list[dict]:
    score_rows = read_jsonl(score_file)
    score_map = {
        (row["dataset_key"], int(row["question_id"]), int(row["item_index"])): row for row in score_rows
    }
    rows: list[dict] = []
    for dataset_key in sorted({key[0] for key in score_map}):
        if dataset_filter is not None and dataset_key not in dataset_filter:
            continue
        by_qid = sorted({key[1] for key in score_map if key[0] == dataset_key})
        for qid in by_qid:
            for item_index, record in enumerate(iter_records(dataset_key, [qid])):
                score = score_map.get((dataset_key, qid, item_index))
                if score is None:
                    continue
                local_seed = seed + qid * 100_003 + item_index * 997 + sum(map(ord, dataset_key))
                features, text, extras = aggregate_answers(
                    record.answers, sample_size=sample_size, rng=np.random.default_rng(local_seed)
                )
                item = {
                    "dataset_key": dataset_key,
                    "question_id": qid,
                    "item_index": item_index,
                    "split": score["split"],
                    "features": features,
                    "text": text,
                    "q1": str(score.get("q1", record.q1)),
                    **extras,
                }
                missing_target = False
                for target_name in target_names:
                    value = score.get(target_name)
                    if value is None and target_name == "asr":
                        value = record.asr
                    if value is None and target_name == "alr":
                        value = record.alr
                    if value is None:
                        missing_target = True
                        break
                    item[target_name] = float(value)
                if not missing_target:
                    rows.append(item)
                if max_rows is not None and len(rows) >= max_rows:
                    return rows
    return rows


def shuffle_response_views(rows: list[dict], seed: int) -> list[dict]:
    """在同攻击方案、同问题内打乱回答视图，保留 q1 与所有目标不变。"""

    rng = np.random.default_rng(seed)
    by_group: dict[tuple[str, int], list[int]] = {}
    for index, row in enumerate(rows):
        by_group.setdefault((row["dataset_key"], row["question_id"]), []).append(index)
    shuffled = [dict(row) for row in rows]
    response_fields = ("features", "text", "sampled_answers", "sample_long_ratio", "sample_refusal_ratio")
    for indices in by_group.values():
        sources = rng.permutation(indices)
        for destination, source in zip(indices, sources.tolist()):
            for field in response_fields:
                shuffled[destination][field] = rows[source][field]
    return shuffled


def fit_handcrafted(
    train: list[dict],
    test: list[dict],
    target: str,
    alpha: float,
    base_feature_indices: list[int] | None = None,
) -> tuple[np.ndarray, dict]:
    x_train = np.stack([row["features"] for row in train])
    x_test = np.stack([row["features"] for row in test])
    y_train = np.asarray([row[target] for row in train])
    names = [f"{stat}_{name}" for stat in ("均值", "标准差", "最小值", "最大值") for name in FEATURE_NAMES]
    if base_feature_indices is not None:
        indices = [offset * len(FEATURE_NAMES) + index for offset in range(4) for index in base_feature_indices]
        x_train = x_train[:, indices]
        x_test = x_test[:, indices]
        names = [names[index] for index in indices]
    scaler = StandardScaler().fit(x_train)
    model = Ridge(alpha=alpha).fit(scaler.transform(x_train), y_train)
    pred = model.predict(scaler.transform(x_test))
    largest = np.argsort(np.abs(model.coef_))[-12:][::-1]
    return pred, {"主要系数": [{"特征": names[i], "系数": float(model.coef_[i])} for i in largest]}


def tfidf_text(row: dict, view: str) -> str:
    if view == "response":
        return row["text"]
    if view == "q1":
        return row["q1"]
    if view == "joint":
        return f"<指令>\n{row['q1']}\n<回答>\n{row['text']}"
    raise ValueError(f"未知文本视图: {view}")


def build_tfidf_matrices(train: list[dict], test: list[dict], max_features: int, view: str):
    """为一个文本视图只拟合一次词表，供多个目标复用。"""

    union = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=max_features // 2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=4, max_features=max_features // 2, sublinear_tf=True)),
        ]
    )
    x_train = union.fit_transform([tfidf_text(row, view) for row in train])
    x_test = union.transform([tfidf_text(row, view) for row in test])
    return x_train, x_test


def main() -> None:
    parser = argparse.ArgumentParser(description="回答内容、攻击指标与伪安全结构的关联分析。")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-sizes", default="1,2,4,8,20")
    parser.add_argument("--targets", default="pseudo_score,asr,alr")
    parser.add_argument("--methods", default="handcrafted")
    parser.add_argument(
        "--text-views",
        default="response",
        help="TF-IDF 输入视图，逗号分隔：response、q1、joint。",
    )
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-datasets", default=None, help="逗号分隔；默认使用 scores 中全部训练集。")
    parser.add_argument("--test-datasets", default=None, help="逗号分隔；默认使用 scores 中全部测试集。")
    parser.add_argument(
        "--shuffle-responses-within-question",
        action="store_true",
        help="同攻击方案、同问题内打乱回答归属，用作负对照。",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_names = [x.strip() for x in args.targets.split(",") if x.strip()]
    train_datasets = {x.strip() for x in args.train_datasets.split(",") if x.strip()} if args.train_datasets else None
    test_datasets = {x.strip() for x in args.test_datasets.split(",") if x.strip()} if args.test_datasets else None
    dataset_filter = (train_datasets or set()) | (test_datasets or set()) or None
    all_results: dict[str, dict] = {}
    for sample_size in [int(x) for x in args.sample_sizes.split(",") if x.strip()]:
        rows = load_rows(args.scores, sample_size, args.seed, args.max_rows, target_names, dataset_filter)
        if args.shuffle_responses_within_question:
            rows = shuffle_response_views(rows, args.seed + sample_size)
        train = [
            row for row in rows
            if row["split"] == "train" and (train_datasets is None or row["dataset_key"] in train_datasets)
        ]
        test = [
            row for row in rows
            if row["split"] == "test" and (test_datasets is None or row["dataset_key"] in test_datasets)
        ]
        if not train or not test:
            raise ValueError("训练集或测试集为空，请检查 train/test dataset 过滤条件。")
        groups = [(row["dataset_key"], row["question_id"]) for row in test]
        sample_result = {
            "训练记录数": len(train),
            "测试记录数": len(test),
            "测试采样长回答比例": float(np.mean([row["sample_long_ratio"] for row in test])),
            "测试采样拒绝开头比例": float(np.mean([row["sample_refusal_ratio"] for row in test])),
            "目标": {},
        }
        tfidf_matrices = {}
        if "tfidf" in args.methods:
            for view in [x.strip() for x in args.text_views.split(",") if x.strip()]:
                tfidf_matrices[view] = build_tfidf_matrices(train, test, args.max_features, view)
        for target in target_names:
            y_train = np.asarray([row[target] for row in train])
            y_test = np.asarray([row[target] for row in test])
            target_result = {}
            if "handcrafted" in args.methods:
                feature_sets = {
                    "仅长度_Ridge": [0, 1, 4],
                    "仅拒绝形式_Ridge": [5, 6, 7],
                    "内容风格_去长度拒绝_Ridge": [2, 3, 8, 9, 10, 11, 12, 13],
                    "全部手工特征_Ridge": None,
                }
                for method_name, indices in feature_sets.items():
                    pred, details = fit_handcrafted(train, test, target, args.alpha, indices)
                    target_result[method_name] = {
                        "评估": sliced_metrics(test, y_test, pred, groups),
                        **details,
                    }
            if "tfidf" in args.methods:
                for view, (x_train, x_test) in tfidf_matrices.items():
                    pred = Ridge(alpha=args.alpha, solver="lsqr").fit(x_train, y_train).predict(x_test)
                    target_result[f"TFIDF_Ridge_{view}"] = {
                        "评估": sliced_metrics(test, y_test, pred, groups)
                    }
            sample_result["目标"][target] = target_result
        all_results[str(sample_size)] = sample_result
        print(f"完成 sample_size={sample_size}: train={len(train)}, test={len(test)}", flush=True)

    report = {
        "说明": "特征仅来自回答正文，未使用 toxic/toxicA；训练测试按固定问题划分。",
        "scores": args.scores,
        "methods": args.methods,
        "text_views": args.text_views,
        "shuffle_responses_within_question": args.shuffle_responses_within_question,
        "train_datasets": sorted(train_datasets) if train_datasets else "全部",
        "test_datasets": sorted(test_datasets) if test_datasets else "全部",
        "sample_sizes": args.sample_sizes,
        "results": all_results,
    }
    output = out_dir / "response_safety_structure_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入 {output}")


if __name__ == "__main__":
    main()
