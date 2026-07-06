"""使用 LoRA 序列分类 CompSP 模型进行 pairwise 推理。"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer, LlamaForSequenceClassification

from compsp.data_io import load_json, save_json


def infer_adapter_vocab_size(lora_path: str) -> int | None:
    adapter_dir = Path(lora_path)
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import safe_open

        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.endswith("embed_tokens.weight"):
                    return int(f.get_tensor(key).shape[0])

    bin_path = adapter_dir / "adapter_model.bin"
    if bin_path.exists():
        state = torch.load(bin_path, map_location="cpu")
        for key, value in state.items():
            if key.endswith("embed_tokens.weight"):
                return int(value.shape[0])
    return None


class PairwiseEvaluator:
    def __init__(self, base_model_path: str, lora_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(lora_path)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.model = LlamaForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=2,
            device_map="auto",
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        )
        vocab_size = infer_adapter_vocab_size(lora_path) or len(self.tokenizer)
        self.model.resize_token_embeddings(vocab_size)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()

    def predict_prob(self, text_a: str, text_b: str) -> float:
        prompt = f'Text A: "{text_a or ""}"\n---\nText B: "{text_b or ""}"\n'
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return torch.softmax(logits, dim=1).squeeze()[1].item()

    def matrix(self, records: list[dict], start: int, end: int) -> np.ndarray:
        items = records[start : end + 1]
        out = np.zeros((len(items), len(items)), dtype=float)
        for (i, item_a), (j, item_b) in product(enumerate(items), enumerate(items)):
            out[i, j] = self.predict_prob(item_a.get("q1", ""), item_b.get("q1", ""))
        return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--lora", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=24)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    records = load_json(args.input)
    evaluator = PairwiseEvaluator(args.base_model, args.lora, args.device)
    save_json(evaluator.matrix(records, args.start, args.end).tolist(), args.output)


if __name__ == "__main__":
    main()
