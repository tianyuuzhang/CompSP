"""数据读取以及 JSON/JSONL 辅助函数。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_questions(path: str | Path, field: str | None = None) -> list[str]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            if not data:
                return []
            if isinstance(data[0], str):
                return data
            if field:
                return [str(row[field]) for row in data]
        raise ValueError(f"不支持的 JSON 问题格式: {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not field:
            for candidate in ("goal", "question", "behavior"):
                if rows and candidate in rows[0]:
                    field = candidate
                    break
        if not field:
            raise ValueError("CSV 输入需要指定 --question-field")
        return [row[field] for row in rows]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
