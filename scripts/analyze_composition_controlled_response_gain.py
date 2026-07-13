#!/usr/bin/env python3
"""成分受控的少量回答信息增益实验。

本脚本检验一个比普通前缀/后缀拆分更严格的问题：当观察到的回答成分被固定后，
回答具体文本是否仍能预测未观察后缀 ASR。这样可以排除模型只是利用
``asr_small/alr_small`` 或 unsafe 个数做朴素估计。

三种默认口径：

- ``unsafe_only``：只观察危险回答，前缀 ASR 恒为 1。
- ``safe_long_only``：只观察安全且较长的回答，前缀 ASR 恒为 0，前缀 ALR 恒为 1。
- ``safe3_unsafe1``：固定 3 条安全回答 + 1 条危险回答，前缀 ASR 恒为 0.25。

输入特征只包含攻击者可见文本：q1、被选回答正文，以及由这些文本派生的统计量。
``toxic`` 只用于离线选择回答成分和构造未观察后缀标签，不作为模型输入。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

from compsp.asr_dataset import iter_records
from compsp.response_structure import aggregate_answers, pairwise_accuracy_by_group
from compsp.whitebox.supervised_direction import TEST_QIDS, TRAIN_QIDS, parse_csv_ints
from simulate_text_residual_sequential_asr import clean_text, select_base_features


@dataclass(frozen=True)
class CompositionSpec:
    """固定回答成分配置。"""

    name: str
    safe_needed: int
    unsafe_needed: int
    long_safe_only: bool = False

    @property
    def observed_n(self) -> int:
        return self.safe_needed + self.unsafe_needed


DEFAULT_SPECS = {
    "unsafe_only": CompositionSpec("unsafe_only", safe_needed=0, unsafe_needed=1),
    "safe_long_only": CompositionSpec("safe_long_only", safe_needed=1, unsafe_needed=0, long_safe_only=True),
    "safe3_unsafe1": CompositionSpec("safe3_unsafe1", safe_needed=3, unsafe_needed=1),
}


def answer_is_unsafe(answer: dict) -> bool:
    """回答级 unsafe 判定，仅用于离线标签。"""

    return "unsafe" in str(answer.get("toxic", "")).lower()


def answer_len(answer: dict) -> int:
    """读取回答长度，缺失时退化为正文字符数。"""

    value = answer.get("len")
    if isinstance(value, (int, float)):
        return int(value)
    return len(str(answer.get("A", "")))


def safe_answer_indices(answers: list[dict], long_threshold: int | None = None) -> list[int]:
    """返回安全回答下标，可选要求长回答。"""

    indices = [idx for idx, answer in enumerate(answers) if not answer_is_unsafe(answer)]
    if long_threshold is not None:
        indices = [idx for idx in indices if answer_len(answers[idx]) >= long_threshold]
    return indices


def unsafe_answer_indices(answers: list[dict]) -> list[int]:
    """返回危险回答下标。"""

    return [idx for idx, answer in enumerate(answers) if answer_is_unsafe(answer)]


def choose_indices(
    answers: list[dict],
    spec: CompositionSpec,
    rng: np.random.Generator,
    long_threshold: int,
) -> list[int] | None:
    """按固定成分抽取观察回答下标。"""

    safe_long_threshold = long_threshold if spec.long_safe_only else None
    safe = safe_answer_indices(answers, safe_long_threshold)
    unsafe = unsafe_answer_indices(answers)
    if len(safe) < spec.safe_needed or len(unsafe) < spec.unsafe_needed:
        return None
    chosen: list[int] = []
    if spec.safe_needed:
        chosen.extend(rng.choice(safe, size=spec.safe_needed, replace=False).astype(int).tolist())
    if spec.unsafe_needed:
        chosen.extend(rng.choice(unsafe, size=spec.unsafe_needed, replace=False).astype(int).tolist())
    rng.shuffle(chosen)
    return chosen


def build_rows(
    dataset_keys: list[str],
    question_ids: list[int],
    split_by_qid: dict[int, str],
    spec: CompositionSpec,
    seed: int,
    length_threshold: int,
) -> list[dict]:
    """构造固定成分样本，标签为未观察后缀 ASR/ALR。"""

    rows: list[dict] = []
    for dataset_key in dataset_keys:
        for qid in question_ids:
            for item_index, record in enumerate(iter_records(dataset_key, [qid])):
                answers = [answer for answer in record.answers if isinstance(answer, dict)]
                if len(answers) <= spec.observed_n:
                    continue
                local_seed = seed * 1_000_003 + qid * 1009 + item_index * 917 + sum(map(ord, dataset_key))
                rng = np.random.default_rng(local_seed)
                chosen = choose_indices(answers, spec, rng, length_threshold)
                if chosen is None:
                    continue
                chosen_set = set(chosen)
                observed = [answers[idx] for idx in chosen]
                suffix = [answer for idx, answer in enumerate(answers) if idx not in chosen_set]
                if not suffix:
                    continue
                future_tx = sum(answer_is_unsafe(answer) for answer in suffix)
                future_txl = sum(answer_len(answer) >= length_threshold for answer in suffix)
                prefix_tx = sum(answer_is_unsafe(answer) for answer in observed)
                prefix_txl = sum(answer_len(answer) >= length_threshold for answer in observed)
                features, response_text, extras = aggregate_answers(
                    observed,
                    sample_size=len(observed),
                    rng=np.random.default_rng(local_seed + 17),
                    long_threshold=length_threshold,
                )
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "question_id": qid,
                        "item_index": item_index,
                        "split": split_by_qid.get(qid, "test"),
                        "q1": record.q1,
                        "response_text": response_text,
                        "joint_text": f"指令：{record.q1}\n\n回答：{response_text}",
                        "future_asr": future_tx / len(suffix),
                        "future_alr": future_txl / len(suffix),
                        "future_tx": int(future_tx),
                        "suffix_n": int(len(suffix)),
                        "prefix_tx": int(prefix_tx),
                        "prefix_txl": int(prefix_txl),
                        "prefix_asr_small": prefix_tx / len(observed),
                        "prefix_alr_small": prefix_txl / len(observed),
                        "features": select_base_features(features),
                        **extras,
                    }
                )
    return rows


def rankdata(values: np.ndarray) -> np.ndarray:
    """简单平均秩实现，用于避免额外依赖。"""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 Spearman 相关。"""

    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(y_true), rankdata(y_pred))[0, 1])


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 Pearson 相关。"""

    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def high_risk_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float]:
    """计算高风险二分类排序指标。"""

    labels = (y_true >= threshold).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"ap": float("nan"), "auc": float("nan"), "high_rate": float(labels.mean())}
    return {
        "ap": float(average_precision_score(labels, y_pred)),
        "auc": float(roc_auc_score(labels, y_pred)),
        "high_rate": float(labels.mean()),
    }


def evaluate_predictions(rows: list[dict], predictions: np.ndarray, target_key: str, high_threshold: float) -> dict[str, float]:
    """汇总回归、排序和高风险检出指标。"""

    y = np.asarray([row[target_key] for row in rows], dtype=float)
    groups = [(row["dataset_key"], row["question_id"]) for row in rows]
    metrics = {
        "n": float(len(rows)),
        "target_mean": float(np.mean(y)) if len(y) else float("nan"),
        "mae": float(mean_absolute_error(y, predictions)) if len(y) else float("nan"),
        "pearson": pearson(y, predictions),
        "spearman": spearman(y, predictions),
        "pairwise_acc_delta_0": pairwise_accuracy_by_group(y, predictions, groups, 0.0),
        "pairwise_acc_delta_0.1": pairwise_accuracy_by_group(y, predictions, groups, 0.1),
    }
    metrics.update(high_risk_metrics(y, predictions, high_threshold))
    return metrics


def fit_count_model(train: list[dict], test: list[dict], target_key: str) -> np.ndarray:
    """只使用前缀计数和长度格式统计的基线。"""

    def matrix(rows: list[dict]) -> np.ndarray:
        scalar = np.asarray(
            [
                [row["prefix_tx"], row["prefix_txl"], row["prefix_asr_small"], row["prefix_alr_small"]]
                for row in rows
            ],
            dtype=float,
        )
        feature = np.stack([row["features"] for row in rows])
        return np.concatenate([scalar, feature], axis=1)

    x_train = matrix(train)
    x_test = matrix(test)
    y_train = np.asarray([row[target_key] for row in train], dtype=float)
    scaler = StandardScaler().fit(x_train)
    return Ridge(alpha=10.0).fit(scaler.transform(x_train), y_train).predict(scaler.transform(x_test))


def fit_text_model(
    train: list[dict],
    test: list[dict],
    target_key: str,
    view: str,
    cleaning: str,
    max_features: int,
    seed: int,
) -> np.ndarray:
    """拟合指定文本视图的 TF-IDF Ridge 模型。"""

    train_texts = [clean_text(row[view], cleaning) for row in train]
    test_texts = [clean_text(row[view], cleaning) for row in test]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=max_features, sublinear_tf=True)
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    y_train = np.asarray([row[target_key] for row in train], dtype=float)
    return Ridge(alpha=10.0, solver="lsqr", random_state=seed).fit(x_train, y_train).predict(x_test)


def shuffled_response_rows(rows: list[dict], seed: int) -> list[dict]:
    """在同数据集、同问题内打乱回答文本，保留 q1 和标签。"""

    rng = np.random.default_rng(seed)
    output = [dict(row) for row in rows]
    groups: dict[tuple[str, int], list[int]] = {}
    for index, row in enumerate(output):
        groups.setdefault((row["dataset_key"], row["question_id"]), []).append(index)
    for indices in groups.values():
        texts = [output[index]["response_text"] for index in indices]
        permuted = [texts[int(index)] for index in rng.permutation(len(texts))]
        for index, text in zip(indices, permuted):
            output[index]["response_text"] = text
            output[index]["joint_text"] = f"指令：{output[index]['q1']}\n\n回答：{text}"
    return output


def aggregate_seed_metrics(values: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """对多 seed 指标取均值和样本标准差。"""

    keys = sorted({key for item in values for key in item})
    result: dict[str, dict[str, float]] = {}
    for key in keys:
        arr = np.asarray([item[key] for item in values if key in item and np.isfinite(item[key])], dtype=float)
        result[key] = {
            "mean": float(np.mean(arr)) if len(arr) else float("nan"),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }
    return result


def run_once(args: argparse.Namespace, spec: CompositionSpec, seed: int) -> dict:
    """运行一个 seed。"""

    train_qids = parse_csv_ints(args.train_question_ids, sorted(TRAIN_QIDS))
    test_qids = parse_csv_ints(args.test_question_ids, sorted(TEST_QIDS))
    question_ids = sorted(set(train_qids + test_qids))
    split_by_qid = {qid: "train" for qid in train_qids} | {qid: "test" for qid in test_qids}
    rows = build_rows(args.dataset_keys, question_ids, split_by_qid, spec, seed, args.length_threshold)
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    if len(train) < 10 or len(test) < 10:
        raise RuntimeError(f"{spec.name} seed={seed} 可用样本过少：train={len(train)}, test={len(test)}")

    output = {
        "seed": seed,
        "train_n": len(train),
        "test_n": len(test),
        "target": {},
    }
    for target_key in args.targets:
        predictions = {
            "count_features": fit_count_model(train, test, target_key),
            "q1_only": fit_text_model(train, test, target_key, "q1", args.text_cleaning, args.max_features, seed),
            "response_only": fit_text_model(train, test, target_key, "response_text", args.text_cleaning, args.max_features, seed),
            "joint": fit_text_model(train, test, target_key, "joint_text", args.text_cleaning, args.max_features, seed),
        }
        shuffled_test = shuffled_response_rows(test, seed + 991)
        predictions["response_shuffled"] = fit_text_model(
            train,
            shuffled_test,
            target_key,
            "response_text",
            args.text_cleaning,
            args.max_features,
            seed,
        )
        output["target"][target_key] = {
            name: evaluate_predictions(test, pred, target_key, args.high_threshold)
            for name, pred in predictions.items()
        }
    return output


def write_markdown(report: dict, path: Path) -> None:
    """写中文 Markdown 摘要。"""

    lines = [
        "# 成分受控回答信息增益实验",
        "",
        "本实验固定已观察回答的 unsafe/safe 成分，再预测未观察后缀 ASR/ALR。",
        "若 `response_only` 在该条件下仍优于 `count_features` 和 `response_shuffled`，才说明回答具体文本提供了超出前缀计数的真实信息增益。",
        "",
    ]
    for spec_name, spec_report in report["compositions"].items():
        lines.extend([f"## {spec_name}", ""])
        for target, target_report in spec_report["summary"].items():
            lines.extend([
                f"### 目标：{target}",
                "",
                "| 视图 | n | target mean | MAE | Spearman | pairwise@0.1 | AP | AUC |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for view, metrics in target_report.items():
                lines.append(
                    f"| {view} | {metrics['n']['mean']:.0f} | {metrics['target_mean']['mean']:.3f} | "
                    f"{metrics['mae']['mean']:.3f} | {metrics['spearman']['mean']:.3f} | "
                    f"{metrics['pairwise_acc_delta_0.1']['mean']:.3f} | {metrics['ap']['mean']:.3f} | "
                    f"{metrics['auc']['mean']:.3f} |"
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="成分受控的少量回答信息增益实验。")
    parser.add_argument("--dataset-keys", default="jbb-llama-ofa,jbb-llama-pair,jbb-llama-drattack")
    parser.add_argument("--train-question-ids", default=",".join(map(str, sorted(TRAIN_QIDS))))
    parser.add_argument("--test-question-ids", default=",".join(map(str, sorted(TEST_QIDS))))
    parser.add_argument("--compositions", default="unsafe_only,safe_long_only,safe3_unsafe1")
    parser.add_argument("--targets", default="future_asr")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--length-threshold", type=int, default=500)
    parser.add_argument("--high-threshold", type=float, default=0.5)
    parser.add_argument("--max-features", type=int, default=12000)
    parser.add_argument("--text-cleaning", default="mask_strong_artifacts")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    args.dataset_keys = [key.strip() for key in args.dataset_keys.split(",") if key.strip()]
    args.targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    composition_names = [name.strip() for name in args.compositions.split(",") if name.strip()]
    specs = [DEFAULT_SPECS[name] for name in composition_names]

    report = {
        "说明": "固定观察回答成分后，用 q1/response/joint 等视图预测未观察后缀 ASR/ALR。",
        "datasets": args.dataset_keys,
        "train_questions": parse_csv_ints(args.train_question_ids, sorted(TRAIN_QIDS)),
        "test_questions": parse_csv_ints(args.test_question_ids, sorted(TEST_QIDS)),
        "text_cleaning": args.text_cleaning,
        "length_threshold": args.length_threshold,
        "high_threshold": args.high_threshold,
        "compositions": {},
    }

    for spec in specs:
        seed_outputs = []
        for seed in range(args.seeds):
            seed_outputs.append(run_once(args, spec, seed))
            print(f"完成 composition={spec.name}, seed={seed}", flush=True)
        summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        for target in args.targets:
            views = seed_outputs[0]["target"][target].keys()
            summary[target] = {
                view: aggregate_seed_metrics([item["target"][target][view] for item in seed_outputs])
                for view in views
            }
        report["compositions"][spec.name] = {
            "safe_needed": spec.safe_needed,
            "unsafe_needed": spec.unsafe_needed,
            "long_safe_only": spec.long_safe_only,
            "seeds": seed_outputs,
            "summary": summary,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, output.with_suffix(".md"))
    print(f"JSON 已写入 {output}")
    print(f"Markdown 已写入 {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
