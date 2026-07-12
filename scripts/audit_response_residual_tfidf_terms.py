#!/usr/bin/env python3
"""审计扣除长度/拒绝/格式后的回答文本残差词组。

脚本先用长度、拒绝和格式手工特征拟合目标，再把训练残差作为 TF-IDF Ridge 的监督目标。
导出的高权重词组用于判断“长度/格式之外”的 ASR 信号更像危险主题、替代建议、语气姿态，
还是攻击模板残片。该脚本只读取 q1、回答和数值目标，不读取 toxic/toxicA。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from analyze_response_safety_structure import FEATURE_NAMES, load_rows, tfidf_text
from analyze_response_residual_signal import FEATURE_SETS, select_features


STRUCTURE_RE = re.compile(r"\b(?:start_header_id|end_header_id|eot_id|assistant|user)\b|[#*_`|]+|[-=]{2,}")


def collect_feature_names(union: FeatureUnion) -> list[str]:
    """返回带 word/char 前缀的特征名，方便人工审计。"""

    names: list[str] = []
    for name, transformer in union.transformer_list:
        names.extend([f"{name}:{feature}" for feature in transformer.get_feature_names_out()])
    return names


def audit_text(row: dict, text_view: str, text_cleaning: str, strip_structure: bool) -> str:
    """生成用于审计的文本；可额外去掉 chat 模板残片和 Markdown 标记。"""

    text = tfidf_text(row, text_view, text_cleaning)
    if strip_structure:
        text = STRUCTURE_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 response TF-IDF 对长度/格式残差的高权重词组。")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default="asr")
    parser.add_argument("--sample-size", type=int, default=1)
    parser.add_argument("--text-view", default="response", choices=("response", "q1", "joint"))
    parser.add_argument(
        "--text-cleaning",
        default="mask_refusal_hazard_terms",
        choices=("none", "mask_refusal_terms", "mask_refusal_hazard_terms"),
    )
    parser.add_argument("--train-datasets", required=True, help="逗号分隔的数据集键。")
    parser.add_argument("--feature-set", default="长度拒绝格式", choices=tuple(FEATURE_SETS))
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--alpha-base", type=float, default=10.0)
    parser.add_argument("--alpha-text", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strip-structure",
        action="store_true",
        help="额外去掉 chat 模板残片和 Markdown 标记，只用于诊断具体词法内容。",
    )
    args = parser.parse_args()

    train_datasets = {x.strip() for x in args.train_datasets.split(",") if x.strip()}
    rows = load_rows(
        args.scores,
        sample_size=args.sample_size,
        seed=args.seed,
        max_rows=None,
        target_names=[args.target],
        dataset_filter=train_datasets,
    )
    train = [row for row in rows if row["split"] == "train" and row["dataset_key"] in train_datasets]
    if not train:
        raise ValueError("训练集为空。")

    x_base = select_features(train, FEATURE_SETS[args.feature_set])
    xb = StandardScaler().fit_transform(x_base)
    y = np.asarray([row[args.target] for row in train])
    base_model = Ridge(alpha=args.alpha_base).fit(xb, y)
    residual = y - base_model.predict(xb)

    union = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=args.max_features // 2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=4, max_features=args.max_features // 2, sublinear_tf=True)),
        ]
    )
    x_text = union.fit_transform([audit_text(row, args.text_view, args.text_cleaning, args.strip_structure) for row in train])
    model = Ridge(alpha=args.alpha_text, solver="lsqr").fit(x_text, residual)
    feature_names = collect_feature_names(union)
    coef = np.asarray(model.coef_)
    top_pos = np.argsort(coef)[-args.top_k :][::-1]
    top_neg = np.argsort(coef)[: args.top_k]

    report = {
        "说明": "正系数表示该词组对应更高的长度/格式残差目标；负系数表示更低。仅用于诊断，不代表因果。",
        "scores": args.scores,
        "target": args.target,
        "sample_size": args.sample_size,
        "text_view": args.text_view,
        "text_cleaning": args.text_cleaning,
        "feature_set": args.feature_set,
        "strip_structure": args.strip_structure,
        "train_datasets": sorted(train_datasets),
        "train_records": len(train),
        "alpha_base": args.alpha_base,
        "alpha_text": args.alpha_text,
        "max_features": args.max_features,
        "residual_summary": {
            "mean": float(np.mean(residual)),
            "std": float(np.std(residual)),
            "min": float(np.min(residual)),
            "max": float(np.max(residual)),
        },
        "top_positive": [{"feature": feature_names[int(i)], "coef": float(coef[int(i)])} for i in top_pos],
        "top_negative": [{"feature": feature_names[int(i)], "coef": float(coef[int(i)])} for i in top_neg],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {output}")


if __name__ == "__main__":
    main()
