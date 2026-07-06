#!/usr/bin/env python3
"""构建简单的 harmful-minus-benign 逐层方向。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from compsp.whitebox.activations import extract_last_and_mean, load_causal_lm
from compsp.whitebox.directions import difference_direction


def load_harmful(path: str, limit: int) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list) and data and isinstance(data[0], str):
        return data[:limit]
    raise ValueError(f"不支持的 harmful 问题格式: {path}")


def load_benign(path: str, limit: int) -> list[str]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    field = "goal" if rows and "goal" in rows[0] else next(iter(rows[0]))
    return [row[field] for row in rows[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--harmful", default="/remote-home/zty/data/jailbreak_questions.json")
    parser.add_argument("--benign", default="/remote-home/zty/data/data_benign-behaviors.csv")
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    harmful = load_harmful(args.harmful, args.n)
    benign = load_benign(args.benign, args.n)
    n = min(len(harmful), len(benign), args.n)
    harmful, benign = harmful[:n], benign[:n]

    model, tokenizer = load_causal_lm(args.model)
    harmful_act = extract_last_and_mean(model, tokenizer, harmful, args.batch_size, args.max_length)
    benign_act = extract_last_and_mean(model, tokenizer, benign, args.batch_size, args.max_length)
    directions = difference_direction(harmful_act.last, benign_act.last)
    harmful_proj = (harmful_act.last.float() * directions.unsqueeze(0)).sum(dim=-1).mean(dim=0)
    benign_proj = (benign_act.last.float() * directions.unsqueeze(0)).sum(dim=-1).mean(dim=0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": args.model,
            "n": n,
            "directions": directions.cpu(),
            "harmful_mean_projection": harmful_proj.cpu(),
            "benign_mean_projection": benign_proj.cpu(),
            "layer_separation": (harmful_proj - benign_proj).cpu(),
            "method": "last_token_harmful_minus_benign",
        },
        out,
    )
    best = int(torch.argmax(harmful_proj - benign_proj).item())
    print(f"saved {out}; n={n}; layers={directions.shape[0]}; best_layer={best}")


if __name__ == "__main__":
    main()
