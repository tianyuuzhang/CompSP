#!/usr/bin/env python3
"""自然前缀回答对白盒 H_proxy 的条件增量实验。

本脚本把现有 ``outputs/whitebox/*_whitebox_scores.jsonl`` 中的逐层拒绝方向投影
作为探索性白盒内部状态代理 ``H_proxy``。它不是最终 H 定义，只用于验证等级 1
实验管线：在控制前缀计数、长度/拒绝/格式和 q1 后，response 是否还能提高对白盒
内部坐标的预测。

当前缓存只覆盖 legacy 20 个问题的 PAIR/DrAttack，因此结果只能作为 smoke/探索；
若要形成重要结论，必须先补全 0-79 问题并做 random80 全局重随机复验。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from analyze_natural_prefix_incremental_response_gain import (
    build_rows,
    clean_text,
    dense_matrix,
    pearson,
    spearman,
)
from compsp.whitebox.supervised_direction import TEST_QIDS, TRAIN_QIDS, parse_csv_ints


WHITEBOX_FILES = {
    "jbb-llama-pair": "outputs/whitebox/jbb-llama-pair_whitebox_scores.jsonl",
    "jbb-llama-drattack": "outputs/whitebox/jbb-llama-drattack_whitebox_scores.jsonl",
}


def load_proxy_scores(dataset_keys: list[str], state_key: str) -> dict[tuple[str, int, int], np.ndarray]:
    """读取白盒 proxy 坐标。"""

    scores: dict[tuple[str, int, int], np.ndarray] = {}
    for dataset_key in dataset_keys:
        path = Path(WHITEBOX_FILES.get(dataset_key, ""))
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (row["dataset_key"], int(row["question_id"]), int(row["item_index"]))
                scores[key] = np.asarray(row[state_key], dtype=np.float32)
    return scores


def attach_proxy(rows: list[dict], proxy: dict[tuple[str, int, int], np.ndarray]) -> list[dict]:
    """只保留有白盒 proxy 的样本。"""

    output = []
    for row in rows:
        key = (row["dataset_key"], int(row["question_id"]), int(row["item_index"]))
        if key in proxy:
            item = dict(row)
            item["h_proxy"] = proxy[key]
            output.append(item)
    return output


def build_designs(train: list[dict], test: list[dict], max_features: int, cleaning: str):
    """构造 count+q1 与 count+q1+response 设计矩阵。"""

    scaler = StandardScaler().fit(dense_matrix(train))
    dense_train = csr_matrix(scaler.transform(dense_matrix(train)))
    dense_test = csr_matrix(scaler.transform(dense_matrix(test)))
    q1_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=max(100, max_features // 2), sublinear_tf=True)
    q1_train = q1_vec.fit_transform([row["q1"] for row in train])
    q1_test = q1_vec.transform([row["q1"] for row in test])
    resp_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=max(100, max_features // 2), sublinear_tf=True)
    resp_train = resp_vec.fit_transform([clean_text(row["response_text"], cleaning) for row in train])
    resp_test = resp_vec.transform([clean_text(row["response_text"], cleaning) for row in test])
    return {
        "count_q1": (hstack([dense_train, q1_train]).tocsr(), hstack([dense_test, q1_test]).tocsr()),
        "count_q1_response": (
            hstack([dense_train, q1_train, resp_train]).tocsr(),
            hstack([dense_test, q1_test, resp_test]).tocsr(),
        ),
    }


def vector_cosine(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """逐样本向量 cosine 的平均值。"""

    denom = np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1)
    mask = denom > 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sum(y_true[mask] * y_pred[mask], axis=1) / denom[mask]))


def evaluate_proxy(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """评价多维 proxy 坐标预测。"""

    layer_spearman = [spearman(y_true[:, idx], y_pred[:, idx]) for idx in range(y_true.shape[1])]
    layer_pearson = [pearson(y_true[:, idx], y_pred[:, idx]) for idx in range(y_true.shape[1])]
    finite_s = [value for value in layer_spearman if np.isfinite(value)]
    finite_p = [value for value in layer_pearson if np.isfinite(value)]
    return {
        "layer_n": float(y_true.shape[1]),
        "mean_layer_spearman": float(np.mean(finite_s)) if finite_s else float("nan"),
        "mean_layer_pearson": float(np.mean(finite_p)) if finite_p else float("nan"),
        "last_layer_spearman": float(layer_spearman[-1]),
        "last_layer_pearson": float(layer_pearson[-1]),
        "vector_cosine": vector_cosine(y_true, y_pred),
        "r2_uniform_average": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
    }


def run_once(args: argparse.Namespace, k: int, seed: int, train_qids: list[int], test_qids: list[int]) -> dict:
    """运行一个 k/seed。"""

    split_by_qid = {qid: "train" for qid in train_qids} | {qid: "test" for qid in test_qids}
    question_ids = sorted(set(train_qids + test_qids))
    proxy = load_proxy_scores(args.dataset_keys, args.state_key)
    rows = build_rows(args.dataset_keys, question_ids, split_by_qid, k, seed, args.length_threshold)
    rows = attach_proxy(rows, proxy)
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    if len(train) < 20 or len(test) < 20:
        raise RuntimeError(f"白盒 proxy 样本过少: train={len(train)}, test={len(test)}")
    designs = build_designs(train, test, args.max_features, args.text_cleaning)
    y_train = np.stack([row["h_proxy"] for row in train])
    y_test = np.stack([row["h_proxy"] for row in test])
    report = {"seed": seed, "train_n": len(train), "test_n": len(test), "models": {}}
    for name, (x_train, x_test) in designs.items():
        model = Ridge(alpha=args.alpha, solver="lsqr").fit(x_train, y_train)
        pred = model.predict(x_test)
        report["models"][name] = evaluate_proxy(y_test, pred)
    report["delta_response_minus_base"] = {
        key: report["models"]["count_q1_response"][key] - report["models"]["count_q1"][key]
        for key in report["models"]["count_q1"]
        if key != "layer_n"
    }
    return report


def aggregate(items: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """聚合多 seed。"""

    keys = sorted({key for item in items for key in item})
    out = {}
    for key in keys:
        values = np.asarray([item[key] for item in items if np.isfinite(item[key])], dtype=float)
        out[key] = {
            "mean": float(np.mean(values)) if len(values) else float("nan"),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="自然前缀对白盒 H_proxy 的条件增量实验。")
    parser.add_argument("--dataset-keys", default="jbb-llama-pair,jbb-llama-drattack")
    parser.add_argument("--state-key", choices=["wb_last", "wb_mean"], default="wb_last")
    parser.add_argument("--train-question-ids", default="1,9,10,15,21,30,37,41,42,45,48,50,55,56,65")
    parser.add_argument("--test-question-ids", default="70,72,73,76,78")
    parser.add_argument("--sample-sizes", default="1,2,4")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--length-threshold", type=int, default=500)
    parser.add_argument("--max-features", type=int, default=6000)
    parser.add_argument("--text-cleaning", default="mask_strong_artifacts")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    args.dataset_keys = [key.strip() for key in args.dataset_keys.split(",") if key.strip()]
    train_qids = parse_csv_ints(args.train_question_ids, [])
    test_qids = parse_csv_ints(args.test_question_ids, [])
    report = {
        "说明": "探索性 H_proxy 实验；当前只覆盖 legacy 20 题 PAIR/DrAttack，不作为最终证据。",
        "datasets": args.dataset_keys,
        "state_key": args.state_key,
        "train_questions": train_qids,
        "test_questions": test_qids,
        "sample_sizes": {},
    }
    for k in [int(value) for value in args.sample_sizes.split(",") if value.strip()]:
        seeds = [run_once(args, k, seed, train_qids, test_qids) for seed in range(args.seeds)]
        report["sample_sizes"][str(k)] = {
            "seeds": seeds,
            "summary": {
                "count_q1": aggregate([item["models"]["count_q1"] for item in seeds]),
                "count_q1_response": aggregate([item["models"]["count_q1_response"] for item in seeds]),
                "delta_response_minus_base": aggregate([item["delta_response_minus_base"] for item in seeds]),
            },
        }
        print(f"完成 k={k}", flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 自然前缀对白盒 H_proxy 的条件增量实验",
        "",
        "当前只覆盖 legacy 20 题 PAIR/DrAttack，属于 smoke/探索，不作为最终机制证据。",
        "",
        "| k | 模型 | mean layer Spearman | last layer Spearman | vector cosine | R2 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for k, item in report["sample_sizes"].items():
        for name in ["count_q1", "count_q1_response", "delta_response_minus_base"]:
            metrics = item["summary"][name]
            lines.append(
                f"| {k} | {name} | {metrics['mean_layer_spearman']['mean']:.3f} | "
                f"{metrics['last_layer_spearman']['mean']:.3f} | {metrics['vector_cosine']['mean']:.3f} | "
                f"{metrics['r2_uniform_average']['mean']:.3f} |"
            )
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON 已写入 {out}")
    print(f"Markdown 已写入 {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
