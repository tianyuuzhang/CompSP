"""基于 pairwise 优势矩阵进行 Bradley-Terry 排序。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from compsp.data_io import load_json, save_json


def bradley_terry_mle(matrix: np.ndarray, lr: float = 0.01, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix 必须是方阵")
    n = matrix.shape[0]
    scores = np.zeros(n)
    for _ in range(max_iter):
        grads = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                p_ij = 1.0 / (1.0 + np.exp(-(scores[i] - scores[j])))
                error = matrix[i, j] - p_ij
                grads[i] += error
                grads[j] -= error
        scores += lr * grads
        scores -= scores.mean()
        if np.linalg.norm(grads) * lr < tol:
            break
    return scores


def save_ranking(matrix_path: str | Path, output_path: str | Path, q_id: int, start_id: int) -> None:
    matrix = np.array(load_json(matrix_path), dtype=float)
    scores = bradley_terry_mle(matrix)
    item_ids = list(range(start_id, start_id + len(scores)))
    sorted_items = sorted(zip(scores, item_ids), key=lambda x: x[0], reverse=True)
    save_json(
        {
            "group_info": {
                "q_idx": q_id,
                "item_id_range": f"{item_ids[0]}-{item_ids[-1]}",
                "item_count": len(item_ids),
                "matrix_shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
            },
            "raw_scores_by_item": [
                {"item_id": item_id, "bt_log_score": float(score)}
                for score, item_id in zip(scores, item_ids)
            ],
            "sorted_scores_by_method": {
                "Bradley-Terry (log)": [
                    {"rank": rank + 1, "item_id": item_id, "score": float(score)}
                    for rank, (score, item_id) in enumerate(sorted_items)
                ]
            },
        },
        output_path,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--q-id", type=int, required=True)
    p.add_argument("--start-id", type=int, default=0)
    args = p.parse_args()
    save_ranking(args.matrix, args.output, args.q_id, args.start_id)


if __name__ == "__main__":
    main()
