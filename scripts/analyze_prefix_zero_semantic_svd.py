#!/usr/bin/env python3
"""用 TF-IDF SVD 近似分析 prefix-zero response 的低维语义结构。

这个脚本是 frozen embedding 前的轻量探针：在强词面掩码后的 response 文本上
拟合 TF-IDF，再用 TruncatedSVD 压到少数低维分量，检查这些分量是否仍能预测
后缀高风险。若少数分量已经有效，说明剩余信号具有低维语义/姿态结构，值得
继续用真正的 frozen embedding 或白盒 hidden state 复验。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from compsp.whitebox.supervised_direction import TEST_QIDS, TRAIN_QIDS, parse_csv_ints
from analyze_prefix_zero_hidden_risk import safe_auc, top_fraction_metrics
from simulate_text_residual_sequential_asr import build_rows, clean_text


def fit_text_svd(train: list[dict], test: list[dict], max_features: int, components: int, cleaning: str):
    """拟合强掩码 response 的 TF-IDF 和 SVD。"""

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=max_features,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform([clean_text(row["text"], cleaning) for row in train])
    x_test = vectorizer.transform([clean_text(row["text"], cleaning) for row in test])
    n_components = min(components, max(1, min(x_train.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=20260713)
    z_train = svd.fit_transform(x_train)
    z_test = svd.transform(x_test)
    return vectorizer, svd, z_train, z_test


def top_component_terms(vectorizer: TfidfVectorizer, svd: TruncatedSVD, top_n: int) -> list[dict]:
    """导出每个 SVD 分量的正负向代表词。"""

    names = np.asarray(vectorizer.get_feature_names_out())
    components = []
    for index, weights in enumerate(svd.components_):
        pos = np.argsort(weights)[::-1][:top_n]
        neg = np.argsort(weights)[:top_n]
        components.append(
            {
                "component": index,
                "explained_variance_ratio": float(svd.explained_variance_ratio_[index]),
                "positive_terms": [str(names[i]) for i in pos],
                "negative_terms": [str(names[i]) for i in neg],
            }
        )
    return components


def evaluate_low_dim(z_train: np.ndarray, z_test: np.ndarray, train: list[dict], test: list[dict], high_threshold: float, top_fraction: float) -> dict:
    """评估 SVD 低维分量对后缀高风险的预测能力。"""

    y_train = np.asarray([row["future_asr"] >= high_threshold for row in train], dtype=int)
    y_test = np.asarray([row["future_asr"] >= high_threshold for row in test], dtype=int)
    report: dict[str, dict] = {}
    if len(np.unique(y_train)) >= 2:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        clf.fit(z_train, y_train)
        score = clf.predict_proba(z_test)[:, 1]
        report["svd_logistic"] = summarize_score(y_test, score, top_fraction)
    reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    reg.fit(z_train, np.asarray([row["future_asr"] for row in train], dtype=float))
    report["svd_ridge"] = summarize_score(y_test, reg.predict(z_test), top_fraction)

    single = []
    for index in range(z_test.shape[1]):
        raw = z_test[:, index]
        score = raw if safe_auc(y_test, raw, "ap") >= safe_auc(y_test, -raw, "ap") else -raw
        item = summarize_score(y_test, score, top_fraction)
        item["component"] = index
        single.append(item)
    single.sort(key=lambda item: item["average_precision"], reverse=True)
    report["best_single_component"] = single[0] if single else {}
    report["top_single_components"] = single[:10]
    return report


def summarize_score(y_true: np.ndarray, score: np.ndarray, top_fraction: float) -> dict:
    """汇总一个连续分数的 AUC/AP/top fraction 指标。"""

    return {
        "roc_auc": safe_auc(y_true, score, "roc"),
        "average_precision": safe_auc(y_true, score, "ap"),
        "top_fraction": top_fraction_metrics(y_true, score, top_fraction),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 prefix-zero response 的低维 SVD 语义结构。")
    parser.add_argument("--dataset-keys", default="jbb-llama-ofa,jbb-llama-pair,jbb-llama-drattack")
    parser.add_argument("--train-question-ids", default=",".join(map(str, sorted(TRAIN_QIDS))))
    parser.add_argument("--test-question-ids", default=",".join(map(str, sorted(TEST_QIDS))))
    parser.add_argument("--sample-sizes", default="1,2,4")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--max-features", type=int, default=12000)
    parser.add_argument("--top-terms", type=int, default=12)
    parser.add_argument("--high-threshold", type=float, default=0.5)
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument("--length-threshold", type=int, default=500)
    parser.add_argument("--text-cleaning", default="mask_strong_artifacts")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_keys = [key.strip() for key in args.dataset_keys.split(",") if key.strip()]
    train_qids = parse_csv_ints(args.train_question_ids, sorted(TRAIN_QIDS))
    test_qids = parse_csv_ints(args.test_question_ids, sorted(TEST_QIDS))
    question_ids = sorted(set(train_qids + test_qids))
    split_by_qid = {qid: "train" for qid in train_qids} | {qid: "test" for qid in test_qids}

    report = {
        "说明": "强掩码 response 的 TF-IDF SVD 低维语义探针。",
        "datasets": dataset_keys,
        "components": args.components,
        "max_features": args.max_features,
        "text_cleaning": args.text_cleaning,
        "sample_sizes": {},
    }
    for k in [int(value) for value in args.sample_sizes.split(",") if value.strip()]:
        seed_reports = []
        for seed in range(args.seeds):
            rows = build_rows(dataset_keys, question_ids, k, seed, args.length_threshold, split_by_qid)
            train = [row for row in rows if row["split"] == "train" and row["prefix_zero"]]
            test = [row for row in rows if row["split"] == "test" and row["prefix_zero"]]
            print(f"开始 k={k}, seed={seed}: train={len(train)}, test={len(test)}", flush=True)
            vectorizer, svd, z_train, z_test = fit_text_svd(train, test, args.max_features, args.components, args.text_cleaning)
            seed_report = {
                "records": len(test),
                "positive": int(sum(row["future_asr"] >= args.high_threshold for row in test)),
                "positive_rate": float(np.mean([row["future_asr"] >= args.high_threshold for row in test])),
                "explained_variance_sum": float(np.sum(svd.explained_variance_ratio_)),
                "metrics": evaluate_low_dim(z_train, z_test, train, test, args.high_threshold, args.top_fraction),
                "components": top_component_terms(vectorizer, svd, args.top_terms),
            }
            seed_reports.append(seed_report)
            print(f"完成 k={k}, seed={seed}", flush=True)
        report["sample_sizes"][str(k)] = seed_reports

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Prefix-zero 强掩码 response 的 SVD 语义探针", ""]
    lines.append(f"分量数：{args.components}；最大词项：{args.max_features}；清理口径：`{args.text_cleaning}`。")
    lines.append("")
    lines.extend(["| k | records | positive_rate | EVR sum | svd_ridge AP | svd_ridge recall | best1 AP | best1 recall |", "|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for k, seed_reports in report["sample_sizes"].items():
        def mean(path: list[str]) -> float:
            values = []
            for seed_report in seed_reports:
                cur = seed_report
                for key in path:
                    cur = cur[key]
                values.append(float(cur))
            return float(np.mean(values))
        lines.append(
            f"| {k} | {int(np.mean([r['records'] for r in seed_reports]))} | "
            f"{mean(['positive_rate']):.3f} | {mean(['explained_variance_sum']):.3f} | "
            f"{mean(['metrics', 'svd_ridge', 'average_precision']):.3f} | "
            f"{mean(['metrics', 'svd_ridge', 'top_fraction', 'recall']):.3f} | "
            f"{mean(['metrics', 'best_single_component', 'average_precision']):.3f} | "
            f"{mean(['metrics', 'best_single_component', 'top_fraction', 'recall']):.3f} |"
        )
    lines.append("")
    lines.append("## 首个 seed 的代表分量词项")
    for k, seed_reports in report["sample_sizes"].items():
        lines.append("")
        lines.append(f"### k={k}")
        for component in seed_reports[0]["components"][:6]:
            lines.append(
                f"- C{component['component']} EVR={component['explained_variance_ratio']:.3f}；"
                f"正向：{', '.join(component['positive_terms'][:8])}；"
                f"负向：{', '.join(component['negative_terms'][:8])}"
            )
    md_path = output.with_suffix(".md")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON 已写入 {output}")
    print(f"Markdown 已写入 {md_path}")


if __name__ == "__main__":
    main()
