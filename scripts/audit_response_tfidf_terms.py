#!/usr/bin/env python3
"""审计回答 TF-IDF 嗅探器的高权重词组。

该脚本只读取 q1/回答文本与已有目标分数，不读取 toxic/toxicA。用途是检查
response-only 模型是否主要依赖显式拒绝套话、危险词，或是否仍有其他文体/语义信号。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion

from analyze_response_safety_structure import load_rows, tfidf_text


def collect_feature_names(union: FeatureUnion) -> list[str]:
    """返回带 word/char 前缀的特征名，便于人工审计。"""

    names: list[str] = []
    for name, transformer in union.transformer_list:
        names.extend([f"{name}:{feature}" for feature in transformer.get_feature_names_out()])
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="导出回答 TF-IDF Ridge 的高权重词组。")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default="asr")
    parser.add_argument("--sample-size", type=int, default=1)
    parser.add_argument("--text-view", default="response", choices=("response", "q1", "joint"))
    parser.add_argument(
        "--text-cleaning",
        default="none",
        choices=("none", "mask_refusal_terms", "mask_refusal_hazard_terms"),
    )
    parser.add_argument("--train-datasets", required=True, help="逗号分隔的数据集键。")
    parser.add_argument("--test-datasets", default=None, help="只用于输出记录数，可不填。")
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_datasets = {x.strip() for x in args.train_datasets.split(",") if x.strip()}
    test_datasets = {x.strip() for x in args.test_datasets.split(",") if x.strip()} if args.test_datasets else None
    dataset_filter = train_datasets | (test_datasets or set())
    rows = load_rows(
        args.scores,
        sample_size=args.sample_size,
        seed=args.seed,
        max_rows=None,
        target_names=[args.target],
        dataset_filter=dataset_filter,
    )
    train = [row for row in rows if row["split"] == "train" and row["dataset_key"] in train_datasets]
    if not train:
        raise ValueError("训练集为空。")

    union = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=args.max_features // 2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=4, max_features=args.max_features // 2, sublinear_tf=True)),
        ]
    )
    x_train = union.fit_transform([tfidf_text(row, args.text_view, args.text_cleaning) for row in train])
    y_train = np.asarray([row[args.target] for row in train])
    model = Ridge(alpha=args.alpha, solver="lsqr").fit(x_train, y_train)
    feature_names = collect_feature_names(union)
    coef = np.asarray(model.coef_)
    top_pos = np.argsort(coef)[-args.top_k :][::-1]
    top_neg = np.argsort(coef)[: args.top_k]

    report = {
        "说明": "正系数表示该词组出现时预测目标更高；负系数表示预测目标更低。仅用于诊断，不代表因果。",
        "scores": args.scores,
        "target": args.target,
        "sample_size": args.sample_size,
        "text_view": args.text_view,
        "text_cleaning": args.text_cleaning,
        "train_datasets": sorted(train_datasets),
        "train_records": len(train),
        "alpha": args.alpha,
        "max_features": args.max_features,
        "top_positive": [{"feature": feature_names[int(i)], "coef": float(coef[int(i)])} for i in top_pos],
        "top_negative": [{"feature": feature_names[int(i)], "coef": float(coef[int(i)])} for i in top_neg],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {output}")


if __name__ == "__main__":
    main()
