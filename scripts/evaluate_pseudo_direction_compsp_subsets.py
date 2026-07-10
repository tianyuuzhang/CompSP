#!/usr/bin/env python3
"""按原始 ALR 子集评估伪安全方向 CompSP 模型。

用途：
1. 重新加载已经训练好的 LoRA pairwise 分类器。
2. 在测试 pair 上重算逐样本预测。
3. 统计完整测试集，以及两条指令原始 ALR 都严格位于 (0, 1) 的子集准确率。

这个子集用于排除一种简单解释：模型只学会了区分 ALR=0、ALR=1、
中间值三类，而没有真正拟合伪安全方向诱导的细粒度序关系。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from transformers import AutoTokenizer, DataCollatorWithPadding, LlamaForSequenceClassification, Trainer, TrainingArguments


def _metric_dict(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
    if len(labels) == 0:
        return {
            "n": 0,
            "accuracy": None,
            "f1_macro": None,
            "precision_macro": None,
            "recall_macro": None,
            "auc": None,
            "label_0": 0,
            "label_1": 0,
        }
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    if len(set(labels.tolist())) == 2:
        auc = float(roc_auc_score(labels, probs))
    else:
        auc = None
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "auc": auc,
        "label_0": int((labels == 0).sum()),
        "label_1": int((labels == 1).sum()),
    }


def _bucket(alr: float) -> str:
    if alr <= 0:
        return "0"
    if alr >= 1:
        return "1"
    return "middle"


def build_subset_report(records: list[dict], labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
    masks: dict[str, list[bool]] = {
        "all": [True for _ in records],
        "both_alr_middle": [0 < r["a_alr"] < 1 and 0 < r["b_alr"] < 1 for r in records],
        "one_or_both_alr_boundary": [not (0 < r["a_alr"] < 1 and 0 < r["b_alr"] < 1) for r in records],
        "same_alr_bucket": [_bucket(r["a_alr"]) == _bucket(r["b_alr"]) for r in records],
        "different_alr_bucket": [_bucket(r["a_alr"]) != _bucket(r["b_alr"]) for r in records],
    }

    report = {}
    for name, mask_list in masks.items():
        mask = np.array(mask_list, dtype=bool)
        report[name] = _metric_dict(labels[mask], preds[mask], probs[mask])

    by_pair_bucket: dict[str, dict] = {}
    for left in ["0", "middle", "1"]:
        for right in ["0", "middle", "1"]:
            key = f"a_{left}__b_{right}"
            mask = np.array([_bucket(r["a_alr"]) == left and _bucket(r["b_alr"]) == right for r in records], dtype=bool)
            if int(mask.sum()) > 0:
                by_pair_bucket[key] = _metric_dict(labels[mask], preds[mask], probs[mask])
    report["by_alr_bucket_pair"] = by_pair_bucket
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="/remote-home/model/llama-3.1-8B/Llama-3.1-8B")
    parser.add_argument("--model-dir", required=True, help="包含 lora/ 和 tokenizer/ 的模型输出目录")
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tokenizer_dir = model_dir / "tokenizer"
    lora_dir = model_dir / "lora"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir if tokenizer_dir.exists() else args.base_model)
    model = LlamaForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        use_cache=False,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    elif len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(model, lora_dir)
    model.eval()

    records = json.loads(Path(args.test_file).read_text(encoding="utf-8"))
    data = load_dataset("json", data_files=args.test_file, split="train")

    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    keep_cols = {"text", "label"}
    remove_cols = [col for col in data.column_names if col not in keep_cols]
    if remove_cols:
        data = data.remove_columns(remove_cols)
    tokenized = data.map(preprocess, batched=True)

    training_args = TrainingArguments(
        output_dir=str(Path(args.output_file).parent / "_tmp_predict"),
        per_device_eval_batch_size=args.batch_size,
        report_to=[],
        bf16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    pred = trainer.predict(tokenized)
    labels = pred.label_ids.astype(int)
    logits = pred.predictions
    preds = logits.argmax(axis=-1)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]

    report = {
        "base_model": args.base_model,
        "model_dir": str(model_dir),
        "test_file": args.test_file,
        "说明": "both_alr_middle 表示 pair 中两条指令的原始 ALR 均严格位于 (0,1)。",
        "metrics": build_subset_report(records, labels, preds, probs),
    }
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()