# Prefix-zero 三视图对照说明

更新时间：2026-07-13

## 目的

该实验用于区分 prefix-zero 隐蔽高风险检出信号来自哪里。

prefix-zero 指前 `k` 次回答均未出现 unsafe 的样本。此前二阶段模型显示，在这类“表面安全”的样本中，前缀回答文本可以把后缀高 ASR 样本显著排到前面。但高权重词项同时包含主题词、攻击模板残片和拒绝结构，因此需要进一步比较三种输入视图：

- `response`：只使用前缀回答文本，检验回答内容本身是否携带信号。
- `q1`：只使用原始攻击指令，作为输入攻击文本 baseline。
- `joint`：拼接 `q1` 和前缀回答，检验联合信息是否更强。

若 `response` 明显强于 `q1`，说明模型回答提供了超出攻击指令文本的额外嗅探信息。若 `q1` 已经接近或超过 `response`，则当前信号很可能主要来自题目、攻击模板或指令语义，而不是回答内容泄露的安全结构。

## 运行方式

远端恢复后，在 `/remote-home/zty/tidy_CompSP` 下运行：

```bash
bash scripts/run_prefix_zero_text_view_ablation.sh
```

默认配置：

- 数据：`jbb-llama-ofa,jbb-llama-pair,jbb-llama-drattack`
- 前缀回答数：`k=1,2,4`
- 训练/测试划分：沿用固定 60/20 问题划分
- 目标：后缀 ASR 是否 `>=0.5`
- 文本视图：`response`、`q1`、`joint`
- 预计耗时：每个视图约 `2 分钟`，总计约 `6-8 分钟`

## 输出

输出写入：

```text
outputs/response_safety_structure/sequential_asr_sampling/
```

主要文件：

- `llama_three_attack_prefix_zero_hidden_risk_response_fast.md`
- `llama_three_attack_prefix_zero_hidden_risk_q1_fast.md`
- `llama_three_attack_prefix_zero_hidden_risk_joint_fast.md`

每个报告包含 ROC-AUC、AP、top 20% precision/recall、max F1 和高权重词项。

## 落档口径

结果落档时需要同时报告：

- `response - q1` 的 AP 差值和 top recall 差值。
- `joint` 是否稳定超过两个单独视图。
- 高权重词项是否仍然集中在主题词、模板残片或拒绝结构。
- 若 `q1` 与 `response` 接近，结论应写为“文本可预测隐蔽风险，但回答内容增量仍需更强控制”，不能直接声称回答内容揭示安全方向。
