#!/usr/bin/env python3
"""使用已保存的白盒方向为 q1 prompt 打分。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from compsp.asr_dataset import load_txfilled_file
from compsp.asr_registry import get_asr_spec
from compsp.whitebox.activations import extract_last_and_mean, load_causal_lm
from compsp.whitebox.directions import project


def parse_ids(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--question-ids", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    direction_data = torch.load(args.direction, map_location="cpu")
    directions = direction_data["directions"].float()
    spec = get_asr_spec(args.dataset_key)
    rows = []
    texts = []
    meta = []
    for qid in parse_ids(args.question_ids):
        path = spec.path_for(qid)
        if not path.exists():
            print(f"跳过缺失文件 {path}")
            continue
        records = load_txfilled_file(path)
        for idx, item in enumerate(records):
            q1 = item.get("q1", "")
            if not q1:
                continue
            texts.append(q1)
            meta.append(
                {
                    "dataset_key": args.dataset_key,
                    "question_id": qid,
                    "item_index": idx,
                    "round": item.get("round"),
                    "asr": item.get("tx", 0) / item.get("totQA", 20),
                    "alr": item.get("txl", 0) / item.get("totQA", 20),
                    "tx": item.get("tx", 0),
                    "txl": item.get("txl", 0),
                    "totQA": item.get("totQA", 20),
                    "q1": q1,
                }
            )

    model, tokenizer = load_causal_lm(args.model)
    acts = extract_last_and_mean(model, tokenizer, texts, args.batch_size, args.max_length)
    last_scores = project(acts.last, directions)
    mean_scores = project(acts.mean, directions)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for m, last, mean in zip(meta, last_scores.tolist(), mean_scores.tolist()):
            m["wb_last"] = last
            m["wb_mean"] = mean
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"已写入 {out}; rows={len(meta)}")


if __name__ == "__main__":
    main()
