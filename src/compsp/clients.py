"""OpenAI 兼容 API 的模型客户端封装。"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Iterable

import aiohttp


@dataclass
class ChatClientConfig:
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    base_url: str | None = None
    timeout_s: int = 120
    max_retries: int = 8
    retry_delay_s: float = 3.0
    temperature: float = 1.0


class OpenAICompatibleClient:
    def __init__(self, config: ChatClientConfig):
        self.config = config
        self.api_key = os.environ.get(config.api_key_env)
        self.base_url = config.base_url or os.environ.get(config.base_url_env)
        if not self.api_key:
            raise RuntimeError(f"缺少 API key 环境变量: {config.api_key_env}")
        if not self.base_url:
            raise RuntimeError(f"缺少 base URL 环境变量: {config.base_url_env}")

    async def chat(self, prompt: str, max_tokens: int = 2048) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "n": 1,
            "temperature": self.config.temperature,
            "max_tokens": max_tokens,
        }
        last_error = ""
        for attempt in range(1, self.config.max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=self.config.timeout_s,
                    ) as response:
                        if response.status == 200:
                            body = await response.json()
                            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                            if content:
                                return content
                            last_error = "模型返回内容为空"
                        else:
                            last_error = f"HTTP {response.status}: {await response.text()}"
            except asyncio.TimeoutError:
                last_error = "请求超时"
            except Exception as exc:
                last_error = repr(exc)

            if attempt < self.config.max_retries:
                await asyncio.sleep(self.config.retry_delay_s * attempt)

        return f"Error: {last_error}"

    async def chat_many(self, prompts: Iterable[str], max_tokens: int, concurrency: int = 16) -> list[str]:
        sem = asyncio.Semaphore(concurrency)

        async def guarded(prompt: str) -> str:
            async with sem:
                return await self.chat(prompt, max_tokens=max_tokens)

        return await asyncio.gather(*(guarded(prompt) for prompt in prompts))
