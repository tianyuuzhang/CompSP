#!/usr/bin/env python3
"""专门分析 prefix-zero 样本中的隐蔽高风险信号。

prefix-zero 指前 k 次回答均未被判定为 unsafe 的指令。上一轮序贯增采显示，
如果只按前缀 ASR/hazard 选择，这类样本几乎不会被选中；但安全审计恰恰需要
发现“前几次看似安全、后续仍可能成功”的低频风险。本脚本只在 prefix-zero
子集内训练/评估二阶段模型，目标是后缀 ASR 是否超过阈值。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from compsp.whitebox.supervised_direction import TEST_QIDS, TRAIN_QIDS, parse_csv_ints
from simulate_text_residual_sequential_asr import build_rows, clean_text


def safe_auc(y_true: np.ndarray, score: np.ndarray, kind: str) -> float:
    """在标签只有单类时返回 NaN，避免指标函数报错。"""

    if len(np.unique(y_true)) < 2:
        return float("nan")
    if kind == "roc":
        return float(roc_auc_score(y_true, score))
    if kind == "ap":
        return float(average_precision_score(y_true, score))
    raise ValueError(kind)


def top_fraction_metrics(y_true: np.ndarray, score: np.ndarray, top_fraction: float) -> dict:
    """按分数选择 top fraction，报告 precision/recall。"""

    count = max(1, int(round(len(score) * top_fraction)))
    order = np.argsort(score)[::-1][:count]
    selected_positive = int(y_true[order].sum())
    total_positive = int(y_true.sum())
    return {
        "selected_n": int(count),
        "precision": selected_positive / count if count else float("nan"),
        "recall": selected_positive / total_positive if total_positive else float("nan"),
    }


def row_text(row: dict, text_view: str, cleaning: str) -> str:
    """按 response/q1/joint 视图取文本，并应用同一清理口径。"""

    if text_view == "response":
        text = row["text"]
    elif text_view == "q1":
        text = row.get("q1", "")
    elif text_view == "joint":
        text = f"<指令>\n{row.get('q1', '')}\n<回答>\n{row['text']}"
    else:
        raise ValueError(f"未知文本视图: {text_view}")
    return clean_text(text, cleaning)


def shuffle_response_texts(rows: list[dict], seed: int, mode: str) -> list[dict]:
    """打乱回答文本归属，用作回答内容增量的负对照。

    `within_question_attack` 在同一数据集、同一问题、同一 split 内打乱 `text`，
    保留 q1、标签和手工特征不变。这个对照保留题目/攻击方案边际分布，但破坏
    具体 q1 与具体回答之间的对应关系。
    """

    if mode == "none":
        return rows
    if mode != "within_question_attack":
        raise ValueError(f"未知回答打乱模式: {mode}")
    copied = [dict(row) for row in rows]
    groups: dict[tuple[str, int, str], list[int]] = {}
    for index, row in enumerate(copied):
        key = (row["dataset_key"], int(row["question_id"]), row["split"])
        groups.setdefault(key, []).append(index)
    rng = np.random.default_rng(20260713 + seed * 1009)
    for indices in groups.values():
        if len(indices) <= 1:
            continue
        texts = [copied[index]["text"] for index in indices]
        shuffled = list(rng.permutation(texts))
        for index, text in zip(indices, shuffled):
            copied[index]["text"] = str(text)
    return copied


def build_tfidf(
    train: list[dict],
    test: list[dict],
    max_features: int,
    cleaning: str,
    use_char_ngrams: bool,
    text_view: str,
):
    """为 prefix-zero 样本拟合指定文本视图的 TF-IDF。"""

    if use_char_ngrams:
        vectorizer = FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=max_features // 2, sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=4, max_features=max_features // 2, sublinear_tf=True)),
            ]
        )
    else:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=max_features, sublinear_tf=True)
    x_train = vectorizer.fit_transform([row_text(row, text_view, cleaning) for row in train])
    x_test = vectorizer.transform([row_text(row, text_view, cleaning) for row in test])
    return vectorizer, x_train, x_test


def top_weight_terms(vectorizer, weights: np.ndarray, top_n: int) -> dict:
    """导出文本模型最强的正向和负向词项，便于判断信号来源。"""

    if hasattr(vectorizer, "get_feature_names_out"):
        names = np.asarray(vectorizer.get_feature_names_out())
    else:
        names = []
        for prefix, transformer in vectorizer.transformer_list:
            for feature in transformer.get_feature_names_out():
                names.append(f"{prefix}:{feature}")
        names = np.asarray(names)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    top_pos = np.argsort(weights)[::-1][:top_n]
    top_neg = np.argsort(weights)[:top_n]
    return {
        "positive": [{"term": str(names[idx]), "weight": float(weights[idx])} for idx in top_pos],
        "negative": [{"term": str(names[idx]), "weight": float(weights[idx])} for idx in top_neg],
    }


def fit_scores(
    train: list[dict],
    test: list[dict],
    high_threshold: float,
    max_features: int,
    cleaning: str,
    use_char_ngrams: bool,
    top_terms: int,
    text_view: str,
):
    """拟合手工特征、文本分类器和文本回归器三类分数。"""

    y_train = np.asarray([row["future_asr"] >= high_threshold for row in train], dtype=int)
    x_base_train = np.stack([row["features"] for row in train])
    x_base_test = np.stack([row["features"] for row in test])
    scaler = StandardScaler().fit(x_base_train)
    xb_train = scaler.transform(x_base_train)
    xb_test = scaler.transform(x_base_test)
    scores: dict[str, np.ndarray] = {}
    if len(np.unique(y_train)) >= 2:
        base_clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(xb_train, y_train)
        scores["base_logistic"] = base_clf.predict_proba(xb_test)[:, 1]
    else:
        scores["base_logistic"] = np.zeros(len(test), dtype=float)
    base_reg = Ridge(alpha=10.0).fit(xb_train, np.asarray([row["future_asr"] for row in train], dtype=float))
    scores["base_ridge"] = base_reg.predict(xb_test)

    vectorizer, x_text_train, x_text_test = build_tfidf(train, test, max_features, cleaning, use_char_ngrams, text_view)
    explanations = {}
    if len(np.unique(y_train)) >= 2:
        text_clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear").fit(x_text_train, y_train)
        scores["text_logistic"] = text_clf.predict_proba(x_text_test)[:, 1]
        explanations["text_logistic"] = top_weight_terms(vectorizer, text_clf.coef_[0], top_terms)
    else:
        scores["text_logistic"] = np.zeros(len(test), dtype=float)
    text_reg = Ridge(alpha=10.0, solver="lsqr").fit(x_text_train, np.asarray([row["future_asr"] for row in train], dtype=float))
    scores["text_ridge"] = text_reg.predict(x_text_test)
    explanations["text_ridge"] = top_weight_terms(vectorizer, text_reg.coef_, top_terms)
    return scores, explanations


def evaluate_scores(test: list[dict], scores: dict[str, np.ndarray], high_threshold: float, top_fraction: float) -> dict:
    """评估 prefix-zero 子集高风险检出能力。"""

    y_true = np.asarray([row["future_asr"] >= high_threshold for row in test], dtype=int)
    report = {
        "records": int(len(test)),
        "positive": int(y_true.sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        "strategies": {},
    }
    rng_score = np.arange(len(test), dtype=float)
    np.random.default_rng(20260712).shuffle(rng_score)
    all_scores = {"random": rng_score, **scores}
    for name, score in all_scores.items():
        score = np.asarray(score, dtype=float)
        precision, recall, thresholds = precision_recall_curve(y_true, score)
        report["strategies"][name] = {
            "roc_auc": safe_auc(y_true, score, "roc"),
            "average_precision": safe_auc(y_true, score, "ap"),
            "top_fraction": top_fraction_metrics(y_true, score, top_fraction),
            "max_f1": float(np.nanmax((2 * precision * recall) / np.maximum(precision + recall, 1e-12))),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 prefix-zero 隐蔽高风险二阶段模型。")
    parser.add_argument("--dataset-keys", required=True)
    parser.add_argument("--train-question-ids", default=",".join(map(str, sorted(TRAIN_QIDS))))
    parser.add_argument("--test-question-ids", default=",".join(map(str, sorted(TEST_QIDS))))
    parser.add_argument("--sample-sizes", default="1,2,4")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--high-threshold", type=float, default=0.5)
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument("--length-threshold", type=int, default=500)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--top-terms", type=int, default=30)
    parser.add_argument(
        "--text-view",
        default="response",
        choices=("response", "q1", "joint"),
        help="TF-IDF 输入视图：仅回答、仅原始攻击指令，或二者拼接。",
    )
    parser.add_argument(
        "--response-shuffle",
        default="none",
        choices=("none", "within_question_attack"),
        help="回答文本归属打乱负对照；只影响 response/joint 中的回答文本。",
    )
    parser.add_argument(
        "--text-cleaning",
        default="mask_refusal_hazard_terms",
        choices=("none", "mask_refusal_terms", "mask_refusal_hazard_terms", "mask_strong_artifacts"),
    )
    parser.add_argument("--use-char-ngrams", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_keys = [key.strip() for key in args.dataset_keys.split(",") if key.strip()]
    train_qids = parse_csv_ints(args.train_question_ids, sorted(TRAIN_QIDS))
    test_qids = parse_csv_ints(args.test_question_ids, sorted(TEST_QIDS))
    question_ids = sorted(set(train_qids + test_qids))
    split_by_qid = {qid: "train" for qid in train_qids} | {qid: "test" for qid in test_qids}

    report = {
        "说明": "只在 prefix-zero 样本内训练/评估，目标是后缀 ASR 是否超过阈值。",
        "datasets": dataset_keys,
        "high_threshold": args.high_threshold,
        "top_fraction": args.top_fraction,
        "text_view": args.text_view,
        "response_shuffle": args.response_shuffle,
        "sample_sizes": {},
    }
    for k in [int(value) for value in args.sample_sizes.split(",") if value.strip()]:
        seed_reports = []
        for seed in range(args.seeds):
            rows = build_rows(dataset_keys, question_ids, k, seed, args.length_threshold, split_by_qid)
            train = [row for row in rows if row["split"] == "train" and row["prefix_zero"]]
            test = [row for row in rows if row["split"] == "test" and row["prefix_zero"]]
            if not train or not test:
                raise ValueError("prefix-zero 训练集或测试集为空。")
            if args.response_shuffle != "none":
                train = shuffle_response_texts(train, seed, args.response_shuffle)
                test = shuffle_response_texts(test, seed + 10_000, args.response_shuffle)
            print(f"开始 k={k}, seed={seed}: prefix-zero train={len(train)}, test={len(test)}", flush=True)
            scores, explanations = fit_scores(
                train,
                test,
                args.high_threshold,
                args.max_features,
                args.text_cleaning,
                args.use_char_ngrams,
                args.top_terms,
                args.text_view,
            )
            seed_report = evaluate_scores(test, scores, args.high_threshold, args.top_fraction)
            seed_report["text_explanations"] = explanations
            seed_reports.append(seed_report)
            print(f"完成 k={k}, seed={seed}", flush=True)
        report["sample_sizes"][str(k)] = seed_reports[0] if len(seed_reports) == 1 else seed_reports

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Prefix-zero 隐蔽高风险二阶段模型", ""]
    lines.append(f"高风险阈值：后缀 ASR >= {args.high_threshold}；选择比例：top {args.top_fraction:.0%}。")
    lines.append(f"文本视图：`{args.text_view}`。")
    lines.append(f"回答归属打乱：`{args.response_shuffle}`。")
    lines.append("")
    for k, k_report in report["sample_sizes"].items():
        if isinstance(k_report, list):
            k_report = k_report[0]
        lines.extend([
            f"## 前缀回答数 k={k}",
            "",
            f"prefix-zero 测试记录：{k_report['records']}；高风险数：{k_report['positive']}；比例：{k_report['positive_rate']:.3f}。",
            "",
            "| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for name, result in k_report["strategies"].items():
            top = result["top_fraction"]
            lines.append(
                f"| {name} | {result['roc_auc']:.3f} | {result['average_precision']:.3f} | "
                f"{top['precision']:.3f} | {top['recall']:.3f} | {result['max_f1']:.3f} |"
            )
        lines.append("")
        explanations = k_report.get("text_explanations", {})
        for model_name in ("text_ridge", "text_logistic"):
            if model_name not in explanations:
                continue
            lines.extend([f"### {model_name} 高权重词项", "", "正向词项："])
            lines.append(", ".join(item["term"] for item in explanations[model_name]["positive"][:15]))
            lines.extend(["", "负向词项："])
            lines.append(", ".join(item["term"] for item in explanations[model_name]["negative"][:15]))
            lines.append("")
    md_path = output.with_suffix(".md")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"结果已写入 {output}")
    print(f"报告已写入 {md_path}")


if __name__ == "__main__":
    main()
