#!/usr/bin/env python3
"""校验已注册的 ASR/ALR txfilled 数据集。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compsp.asr_dataset import discover_question_ids, validate_file
from compsp.asr_registry import ASR_DATASETS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-key", action="append", choices=sorted(ASR_DATASETS))
    parser.add_argument("--sample-items", type=int, default=3, help="每个文件检查的条目数；-1 表示检查全部条目")
    parser.add_argument("--max-files", type=int, default=0, help="每个数据集最多检查的文件数；0 表示全部")
    parser.add_argument("--output", help="可选的 JSON 报告输出路径")
    args = parser.parse_args()

    keys = args.dataset_key or sorted(ASR_DATASETS)
    report = {}
    total_errors = 0
    sample_items = None if args.sample_items < 0 else args.sample_items

    for key in keys:
        spec = ASR_DATASETS[key]
        ids = discover_question_ids(spec)
        if args.max_files:
            ids = ids[: args.max_files]
        dataset_errors = []
        files_checked = 0
        files_missing = []
        for question_id in ids:
            path = spec.path_for(question_id)
            if not path.exists():
                files_missing.append(question_id)
                continue
            result = validate_file(path, spec, sample_items=sample_items)
            files_checked += 1
            if result["errors"]:
                dataset_errors.append({"question_id": question_id, **result})
        total_errors += len(dataset_errors) + len(files_missing)
        report[key] = {
            "dataset": spec.dataset,
            "attack": spec.attack,
            "target_model": spec.target_model,
            "path_template": spec.path_template,
            "question_ids_discovered": ids,
            "files_checked": files_checked,
            "files_missing": files_missing,
            "files_with_errors": dataset_errors,
            "notes": spec.notes,
        }
        print(
            f"{key}: 已检查={files_checked} 缺失={len(files_missing)} "
            f"含错误文件数={len(dataset_errors)}"
        )
        for item in dataset_errors[:5]:
            print(f"  Q_{item['question_id']}: {item['errors'][:3]}")
        if files_missing[:5]:
            print(f"  缺失样例: {files_missing[:5]}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入报告: {out}")
    raise SystemExit(1 if total_errors else 0)


if __name__ == "__main__":
    main()
