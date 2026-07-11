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


def load_rows(score_file: str, sample_size: int, seed: int, max_rows: int | None) -> list[dict]:
    score_rows = read_jsonl(score_file)
    score_map = {
        (row["dataset_key"], int(row["question_id"]), int(row["item_index"])): row for row in score_rows
    }
    rows: list[dict] = []
    for dataset_key in sorted({key[0] for key in score_map}):
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
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "question_id": qid,
                        "item_index": item_index,
                        "split": score["split"],
                        "features": features,
                        "text": text,
                        "pseudo_score": float(score["pseudo_score"]),
                        "asr": float(score["asr"]),
                        "alr": float(score["alr"]),
                        **extras,
                    }
                )
                if max_rows is not None and len(rows) >= max_rows:
                    return rows
    return rows


def fit_handcrafted(train: list[dict], test: list[dict], target: str, alpha: float) -> tuple[np.ndarray, dict]:
    x_train = np.stack([row["features"] for row in train])
    x_test = np.stack([row["features"] for row in test])
    y_train = np.asarray([row[target] for row in train])
    scaler = StandardScaler().fit(x_train)
    model = Ridge(alpha=alpha).fit(scaler.transform(x_train), y_train)
    pred = model.predict(scaler.transform(x_test))
    names = [f"{stat}_{name}" for stat in ("均值", "标准差", "最小值", "最大值") for name in FEATURE_NAMES]
    largest = np.argsort(np.abs(model.coef_))[-12:][::-1]
    return pred, {"主要系数": [{"特征": names[i], "系数": float(model.coef_[i])} for i in largest]}


def fit_tfidf(train: list[dict], test: list[dict], target: str, alpha: float, max_features: int) -> np.ndarray:
    union = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=max_features // 2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=4, max_features=max_features // 2, sublinear_tf=True)),
        ]
    )
    x_train = union.fit_transform([row["text"] for row in train])
    x_test = union.transform([row["text"] for row in test])
    y_train = np.asarray([row[target] for row in train])
    return Ridge(alpha=alpha, solver="lsqr").fit(x_train, y_train).predict(x_test)


def main() -> None:
    parser = argparse.ArgumentParser(description="回答内容、攻击指标与伪安全结构的关联分析。")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-sizes", default="1,2,4,8,20")
    parser.add_argument("--targets", default="pseudo_score,asr,alr")
    parser.add_argument("--methods", default="handcrafted")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = {}
    for sample_size in [int(x) for x in args.sample_sizes.split(",") if x.strip()]:
        rows = load_rows(args.scores, sample_size, args.seed, args.max_rows)
        train = [row for row in rows if row["split"] == "train"]
        test = [row for row in rows if row["split"] == "test"]
        groups = [(row["dataset_key"], row["question_id"]) for row in test]
        sample_result = {
            "训练记录数": len(train),
            "测试记录数": len(test),
            "测试采样长回答比例": float(np.mean([row["sample_long_ratio"] for row in test])),
            "测试采样拒绝开头比例": float(np.mean([row["sample_refusal_ratio"] for row in test])),
            "目标": {},
        }
        for target in [x.strip() for x in args.targets.split(",") if x.strip()]:
            y_test = np.asarray([row[target] for row in test])
            target_result = {}
            if "handcrafted" in args.methods:
                pred, details = fit_handcrafted(train, test, target, args.alpha)
                target_result["手工特征_Ridge"] = {**metrics(y_test, pred, groups), **details}
            if "tfidf" in args.methods:
                pred = fit_tfidf(train, test, target, args.alpha, args.max_features)
                target_result["去拒绝前缀_TFIDF_Ridge"] = metrics(y_test, pred, groups)
            sample_result["目标"][target] = target_result
        all_results[str(sample_size)] = sample_result
        print(f"完成 sample_size={sample_size}: train={len(train)}, test={len(test)}", flush=True)

    report = {
        "说明": "特征仅来自回答正文，未使用 toxic/toxicA；训练测试按固定问题划分。",
        "scores": args.scores,
        "methods": args.methods,
        "sample_sizes": args.sample_sizes,
        "results": all_results,
    }
    output = out_dir / "response_safety_structure_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入 {output}")


if __name__ == "__main__":
    main()

