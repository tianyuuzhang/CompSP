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

## 伪安全结构可学习性实验

- `build_pseudo_safety_direction.py`：从 Llama q1 最后一层 hidden state 和 ALR/ASR 学习 Ridge 伪安全方向，并输出每条 q1 的投影。
- `build_pseudo_safety_pairs.py`：按伪方向投影大小关系构造新的 CompSP pairwise 训练/测试数据。
- `train_pseudo_direction_compsp.py`：重新训练 LoRA pairwise 分类模型；不能复用旧 ASR CompSP。
- `run_pseudo_safety_learnability.sh`：一键运行 ALR 主实验，支持 `SMOKE=1` 小样本烟测。
- `launch_pseudo_safety_asr.sh`：以 `METRIC=asr` 后台启动同配置 ASR 辅助实验，并避免重复启动完整结果。
- `analyze_response_safety_structure.py`：分析单条/少量回答内容与 ASR、ALR、白盒伪安全投影的关联；不使用安全判定字段作为特征。
