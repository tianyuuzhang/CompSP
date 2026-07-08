#!/usr/bin/env python3
"""把伪安全方向投影分数转换成 CompSP pairwise 训练/测试数据。"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from compsp.whitebox.supervised_direction import build_pseudo_pairs, parse_dataset_keys, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--min-delta", type=float, default=0.15)
    parser.add_argument("--max-train-pairs-per-question", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = [row for row in read_jsonl(args.scores) if row.get("dataset_key") == args.dataset_key]
    if not rows:
        raise SystemExit(f"没有找到 dataset_key={args.dataset_key} 的分数记录")

    out_dir = Path(args.output_dir) / args.dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    train_pairs = build_pseudo_pairs(
        rows,
        "train",
        min_delta=args.min_delta,
        max_pairs_per_question=args.max_train_pairs_per_question,
        seed=args.seed,
    )
    test_pairs = build_pseudo_pairs(rows, "test", min_delta=args.min_delta, max_pairs_per_question=None, seed=args.seed)

    for name, data in (("train", train_pairs), ("test", test_pairs)):
        path = out_dir / f"{name}_pairs.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def describe(data: list[dict]) -> dict:
        by_q = Counter(item["questionID"] for item in data)
        labels = Counter(item["label"] for item in data)
        return {
            "pairs": len(data),
            "questions": len(by_q),
            "min_pairs_per_question": min(by_q.values()) if by_q else 0,
            "max_pairs_per_question": max(by_q.values()) if by_q else 0,
            "label_counts": {str(k): int(v) for k, v in sorted(labels.items())},
        }

    summary = {
        "dataset_key": args.dataset_key,
        "score_file": args.scores,
        "min_delta": args.min_delta,
        "train": describe(train_pairs),
        "test": describe(test_pairs),
        "train_file": str(out_dir / "train_pairs.json"),
        "test_file": str(out_dir / "test_pairs.json"),
    }
    (out_dir / "pairs_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
