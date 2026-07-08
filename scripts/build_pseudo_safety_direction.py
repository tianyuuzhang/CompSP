#!/usr/bin/env python3
"""学习伪安全方向并为 q1 输出投影分数。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from compsp.whitebox.activations import extract_layer_last, load_causal_lm
from compsp.whitebox.supervised_direction import (
    DEFAULT_DATASET_KEYS,
    TEST_QIDS,
    TRAIN_QIDS,
    collect_direction_records,
    fit_ridge_direction,
    parse_csv_ints,
    parse_dataset_keys,
    project_with_ridge,
    save_direction,
    safe_corr,
    write_jsonl,
)


def parse_alphas(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/remote-home/model/llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset-keys", default=",".join(DEFAULT_DATASET_KEYS))
    parser.add_argument("--metric", choices=["alr", "asr"], default="alr")
    parser.add_argument("--train-qids", default=",".join(map(str, TRAIN_QIDS)))
    parser.add_argument("--test-qids", default=",".join(map(str, TEST_QIDS)))
    parser.add_argument("--alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-items-per-question", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="mixed_alr")
    args = parser.parse_args()

    dataset_keys = parse_dataset_keys(args.dataset_keys)
    train_qids = parse_csv_ints(args.train_qids, TRAIN_QIDS)
    test_qids = parse_csv_ints(args.test_qids, TEST_QIDS)
    rows = collect_direction_records(dataset_keys, train_qids, test_qids, args.max_items_per_question)
    if not rows:
        raise SystemExit("没有收集到 q1 记录")

    texts = [row.q1 for row in rows]
    model, tokenizer = load_causal_lm(args.model)
    activations = extract_layer_last(model, tokenizer, texts, args.batch_size, args.max_length, layer_index=-1)
    x = activations.numpy().astype("float32")
    y = np.array([getattr(row, args.metric) for row in rows], dtype="float32")
    train_mask = np.array([row.split == "train" for row in rows], dtype=bool)
    test_mask = np.array([row.split == "test" for row in rows], dtype=bool)
    groups = np.array([row.question_id for row in rows], dtype=int)

    ridge, report, _ = fit_ridge_direction(x[train_mask], y[train_mask], groups[train_mask], parse_alphas(args.alphas))
    pseudo = project_with_ridge(ridge, x)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_rows = []
    for row, score in zip(rows, pseudo.tolist()):
        item = asdict(row)
        item["pseudo_score"] = float(score)
        item["direction_metric"] = args.metric
        item["direction_prefix"] = args.prefix
        score_rows.append(item)

    report = report.__class__(
        metric=args.metric,
        selected_alpha=report.selected_alpha,
        train_rows=int(train_mask.sum()),
        test_rows=int(test_mask.sum()),
        train_questions=sorted(set(train_qids)),
        test_questions=sorted(set(test_qids)),
        dataset_keys=dataset_keys,
        cv_results=report.cv_results,
    )
    summary = {
        "report": asdict(report),
        "test_pearson": safe_corr(pseudo[test_mask], y[test_mask], "pearson"),
        "test_spearman": safe_corr(pseudo[test_mask], y[test_mask], "spearman"),
        "train_pearson": safe_corr(pseudo[train_mask], y[train_mask], "pearson"),
        "train_spearman": safe_corr(pseudo[train_mask], y[train_mask], "spearman"),
        "score_file": str(out_dir / f"{args.prefix}_scores.jsonl"),
        "direction_file": str(out_dir / f"{args.prefix}_direction.pt"),
    }

    write_jsonl(out_dir / f"{args.prefix}_scores.jsonl", score_rows)
    save_direction(out_dir / f"{args.prefix}_direction.pt", ridge, report, {"summary": summary})
    (out_dir / f"{args.prefix}_direction_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
