#!/usr/bin/env python3
"""训练新的 CompSP 式伪安全方向 pairwise 分类器。

本脚本不加载任何旧 CompSP LoRA。训练目标来自伪安全方向投影序关系，
用于验证该白盒线性结构是否能被 pairwise 文本代理学习。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from transformers import AutoTokenizer, DataCollatorWithPadding, LlamaForSequenceClassification, Trainer, TrainingArguments


def compute_metrics(pred):
    labels = pred.label_ids.astype(int)
    logits = pred.predictions
    preds = logits.argmax(axis=-1)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    precision, recall, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    out = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_macro),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "label_0": int((labels == 0).sum()),
        "label_1": int((labels == 1).sum()),
    }
    if len(set(labels.tolist())) == 2:
        out["auc"] = float(roc_auc_score(labels, probs))
    else:
        out["auc"] = float("nan")
    return out


def load_pair_dataset(path: str):
    data = load_dataset("json", data_files=path, split="train")
    keep_cols = {"text", "label"}
    remove_cols = [col for col in data.column_names if col not in keep_cols]
    if remove_cols:
        data = data.remove_columns(remove_cols)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="/remote-home/model/llama-3.1-8B/Llama-3.1-8B")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = LlamaForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        use_cache=False,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    )
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    train_data = load_pair_dataset(args.train_file)
    test_data = load_pair_dataset(args.test_file)

    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_data = train_data.map(preprocess, batched=True).shuffle(seed=args.seed)
    test_data = test_data.map(preprocess, batched=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    initial_metrics = trainer.evaluate()
    train_result = trainer.train()
    final_metrics = trainer.evaluate()
    model.save_pretrained(output_dir / "lora")
    tokenizer.save_pretrained(output_dir / "tokenizer")

    summary = {
        "base_model": args.base_model,
        "train_file": args.train_file,
        "test_file": args.test_file,
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "initial_metrics": initial_metrics,
        "train_metrics": train_result.metrics,
        "final_metrics": final_metrics,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
