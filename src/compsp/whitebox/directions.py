"""安全/拒绝方向的构造与投影。"""

from __future__ import annotations

import torch


def difference_direction(harmful_last: torch.Tensor, benign_last: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """返回归一化后的逐层 harmful-minus-benign 方向。

    输入形状为 `[n, layers, hidden]`；输出形状为 `[layers, hidden]`。
    """

    direction = harmful_last.mean(dim=0) - benign_last.mean(dim=0)
    return direction / direction.norm(dim=-1, keepdim=True).clamp_min(eps)


def project(activations: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """将激活 `[n, layers, hidden]` 投影到方向 `[layers, hidden]` 上。"""

    return (activations.float() * directions.float().unsqueeze(0)).sum(dim=-1)
