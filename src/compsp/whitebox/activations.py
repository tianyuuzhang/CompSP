"""Llama 类因果语言模型的 hidden state 提取工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ActivationBatch:
    last: torch.Tensor
    mean: torch.Tensor


def load_causal_lm(model_path: str, device_map: str = "auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def format_user_prompt(tokenizer, text: str, use_chat_template: bool = True) -> str:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


@torch.no_grad()
def extract_last_and_mean(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 4,
    max_length: int = 512,
    use_chat_template: bool = True,
) -> ActivationBatch:
    last_chunks: list[torch.Tensor] = []
    mean_chunks: list[torch.Tensor] = []
    model_device = next(model.parameters()).device

    for start in range(0, len(texts), batch_size):
        batch_texts = [format_user_prompt(tokenizer, text, use_chat_template) for text in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(model_device)
        attention_mask = encoded["attention_mask"].to(model_device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        lengths = attention_mask.sum(dim=1) - 1
        layer_last = []
        layer_mean = []
        mask = attention_mask.unsqueeze(-1)
        for hidden in outputs.hidden_states:
            hidden_f = hidden.float()
            layer_last.append(hidden_f[torch.arange(hidden_f.shape[0], device=hidden_f.device), lengths])
            layer_mean.append((hidden_f * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1))
        last_chunks.append(torch.stack(layer_last, dim=1).cpu())
        mean_chunks.append(torch.stack(layer_mean, dim=1).cpu())
        del outputs, input_ids, attention_mask, encoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return ActivationBatch(last=torch.cat(last_chunks, dim=0), mean=torch.cat(mean_chunks, dim=0))


@torch.no_grad()
def extract_layer_last(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 4,
    max_length: int = 512,
    layer_index: int = -1,
    use_chat_template: bool = True,
) -> torch.Tensor:
    """只提取指定层最后一个有效 token 的 hidden state。

    伪安全方向实验只需要最后一层最后 token 表示；相比 `extract_last_and_mean`，
    这个函数不会把所有层堆叠到 CPU，适合 1 万条以上 q1 的批量抽取。
    输出形状为 `[n, hidden]`。
    """

    chunks: list[torch.Tensor] = []
    model_device = next(model.parameters()).device

    for start in range(0, len(texts), batch_size):
        batch_texts = [format_user_prompt(tokenizer, text, use_chat_template) for text in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(model_device)
        attention_mask = encoded["attention_mask"].to(model_device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        lengths = attention_mask.sum(dim=1) - 1
        hidden = outputs.hidden_states[layer_index].float()
        last = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
        chunks.append(last.cpu())
        del outputs, input_ids, attention_mask, encoded, hidden, last
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(chunks, dim=0)


def batched(iterable: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]
