#!/usr/bin/env python3
"""用 Llama 回答文本 hidden embedding 嗅探输入端伪安全投影。

这是白盒回答表征实验：回答先删除常见拒绝前缀，再作为纯文本输入 Llama；提取最后层
最后 token 表示。Ridge 只在固定 60 个训练问题上拟合，20 个测试问题只用于评估。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from compsp.asr_dataset import iter_records
from compsp.response_structure import answer_view, pairwise_accuracy_by_group
from compsp.whitebox.activations import extract_layer_last, load_causal_lm
from compsp.whitebox.supervised_direction import read_jsonl, write_jsonl


def safe_corr(left: np.ndarray, right: np.ndarray, method: str) -> float:
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    result = pearsonr(left, right) if method == "pearson" else spearmanr(left, right)
    return float(result.statistic)


def collect_rows(score_file: str, seed: int) -> list[dict]:
    score_rows = read_jsonl(score_file)
    score_map = {
        (row["dataset_key"], int(row["question_id"]), int(row["item_index"])): row for row in score_rows
    }
    rows: list[dict] = []
    for dataset_key in sorted({key[0] for key in score_map}):
        for qid in sorted({key[1] for key in score_map if key[0] == dataset_key}):
            for item_index, record in enumerate(iter_records(dataset_key, [qid])):
                score = score_map.get((dataset_key, qid, item_index))
                if score is None or not record.answers:
                    continue
                local_seed = seed + qid * 100_003 + item_index * 997 + sum(map(ord, dataset_key))
                rng = np.random.default_rng(local_seed)
                answer = record.answers[int(rng.integers(0, len(record.answers)))]
                text = answer_view(str(answer.get("A", ""))).substantive_text
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "question_id": qid,
                        "item_index": item_index,
                        "split": score["split"],
                        "text": text,
                        "pseudo_score": float(score["pseudo_score"]),
                        "asr": float(score["asr"]),
                        "alr": float(score["alr"]),
                    }
                )
    return rows


def evaluate_slices(rows: list[dict], target: np.ndarray, pred: np.ndarray) -> dict:
    result = {}
    slices = {"合并": np.ones(len(rows), dtype=bool)}
    for dataset_key in sorted({row["dataset_key"] for row in rows}):
        slices[dataset_key] = np.asarray([row["dataset_key"] == dataset_key for row in rows], dtype=bool)
    for name, mask in slices.items():
        y = target[mask]
        p = pred[mask]
        groups = [
            (row["dataset_key"], row["question_id"])
            for row, keep in zip(rows, mask.tolist())
            if keep
        ]
        result[name] = {
            "n": int(mask.sum()),
            "pearson": safe_corr(p, y, "pearson"),
            "spearman": safe_corr(p, y, "spearman"),
            "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p)),
            "同题序关系准确率": pairwise_accuracy_by_group(y, p, groups),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="回答 hidden embedding 与安全结构关联分析。")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--model", default="/remote-home/model/llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args.scores, args.seed)
    model, tokenizer = load_causal_lm(args.model)
    embeddings = extract_layer_last(
        model,
        tokenizer,
        [row["text"] for row in rows],
        batch_size=args.batch_size,
        max_length=args.max_length,
        layer_index=-1,
        use_chat_template=False,
        show_progress=True,
    ).numpy().astype("float32")

    np.save(out_dir / "response_last_layer_embeddings.npy", embeddings)
    write_jsonl(out_dir / "response_embedding_rows.jsonl", rows)
    train_mask = np.asarray([row["split"] == "train" for row in rows], dtype=bool)
    test_mask = ~train_mask
    test_rows = [row for row in rows if row["split"] == "test"]
    report = {
        "说明": "单回答去拒绝前缀后，以纯文本提取 Llama 最后一层最后 token 表示。",
        "model": args.model,
        "alpha": args.alpha,
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "targets": {},
    }
    for target_name in ("pseudo_score", "asr", "alr"):
        y = np.asarray([row[target_name] for row in rows], dtype=np.float32)
        ridge = Ridge(alpha=args.alpha, solver="lsqr").fit(embeddings[train_mask], y[train_mask])
        pred = ridge.predict(embeddings[test_mask])
        report["targets"][target_name] = evaluate_slices(test_rows, y[test_mask], pred)
        np.save(out_dir / f"{target_name}_ridge_coef.npy", ridge.coef_.astype("float32"))
    output = out_dir / "response_hidden_embedding_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

