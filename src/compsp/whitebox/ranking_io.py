"""从 data/saverk 读取 CompSP Bradley-Terry 分数。"""

from __future__ import annotations

import json
from pathlib import Path


SAVERK_DIRS = {
    "jbb-llama-pair": "data/saverk/save_jbb2100_pair_llama",
    "jbb-llama-drattack": "data/saverk/save_jbb4100_dr_ins_llama",
    "jbb-4omini-pair": "data/saverk/save_jbb3100_pair_4omini",
    "jbb-4omini-drattack": "data/saverk/save_jbb1100_dr_ins_4omini",
    "jbb-qwen-pair": "data/saverk/save_jbb6100_pair_qwen",
    "jbb-qwen-drattack": "data/saverk/save_jbb5100_dr_ins_qwen",
}


def load_btl_scores(project_root: str | Path, dataset_key: str, question_id: int) -> dict[int, float]:
    root = Path(project_root)
    rel = SAVERK_DIRS.get(dataset_key)
    if rel is None:
        raise KeyError(f"没有找到 {dataset_key} 对应的 saverk 映射")
    q_dir = root / rel / f"Q_{question_id}"
    scores: dict[int, float] = {}
    for path in sorted(q_dir.glob(f"Q_{question_id}_*_ranking_scores.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("raw_scores_by_item", []):
            scores[int(item["item_id"])] = float(item["bt_log_score"])
    return scores
