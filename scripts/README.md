# scripts

围绕 `src/compsp` 的轻量命令行包装层。

- `run_ofa.py`：运行 `init`、`q1`、`answers`、`judge` 或 `all` OFA 阶段。
- `run_pairwise_matrix.py`：加载 CompSP LoRA 分类器并写出 pairwise 优势矩阵。
- `run_bt_rank.py`：将 pairwise 矩阵转换为 Bradley-Terry 排序分数。
- `compute_attack_scheme_metrics.py`：计算 OFA、Pair、DrAttack 在固定四段排序下相对随机排序的 IASR/FASC 提升。
- `validate_asr_data.py`：检查已注册的 ASR/ALR `txfilled` 文件，并可写出 JSON 校验报告。
- `build_refusal_direction.py`：构建逐层的 harmful-minus-benign 方向。
- `score_whitebox_q1.py`：用已保存方向对 `txfilled` 的 `q1` 提示词打分。
- `analyze_whitebox_compsp_alignment.py`：将白盒分数与 `data/saverk` 中的 BTL 排名合并。

在 `/remote-home/zty/tidy_CompSP` 下运行时请先设置：

```bash
export PYTHONPATH=$PWD/src
```
