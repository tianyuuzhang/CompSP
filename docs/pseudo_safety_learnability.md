# 伪安全结构可学习性实验

更新时间：2026-07-08

## 目标

本实验验证一个比“是否学到单一安全向量”更宽的命题：由白盒 Llama 最后一层表示和行为监督定义出的线性伪安全结构，能否被 CompSP 式 pairwise 文本代理学习。

这里的“伪安全方向”不是论文中已经证明的真实安全向量，而是实验定义：

```text
q1 -> Llama-3.1-8B-Instruct 最后一层最后 token hidden state -> Ridge(ALR/ASR) -> 一维投影
```

如果新训练的 CompSP 模型能在留出问题上准确预测两个 q1 的投影大小关系，就说明这种白盒表示中的安全相关序结构可以通过 pairwise 文本监督被学习。

## 固定数据划分

测试题沿用旧论文 20 个测试问题：

```text
[9, 30, 73, 10, 37, 15, 42, 1, 50, 41, 45, 65, 70, 72, 21, 55, 56, 78, 48, 76]
```

训练题固定为 `0~79` 中排除上述测试题后的 60 个问题。主实验不使用 `80~99`，避免改变前期论文划分。

每个攻击方案的数据规模约为：

- 训练：`60 * 100 = 6000` 条 q1。
- 测试：`20 * 100 = 2000` 条 q1。

主数据集：

- `jbb-llama-ofa`
- `jbb-llama-pair`
- `jbb-llama-drattack`

## 为什么不能复用旧 CompSP

旧 CompSP 模型与目标模型、攻击方案和训练指标绑定。之前已有模型主要学习的是各目标模型上的 ASR 序关系，例如 Qwen-ASR 或 Llama-ASR。

本实验的训练目标已经变成：

```text
pseudo_score(q1_i) > pseudo_score(q1_j)
```

其中 `pseudo_score` 是白盒隐藏层线性方向投影。因此旧 CompSP LoRA、旧 `data/saverk` BTL 分数和旧 ASR 排序结果都不能作为本实验主证据。若后续需要 BTL，也必须用新训练模型重新计算 pairwise matrix 和 BTL。

## 代码入口

### 1. 学习伪安全方向并输出投影

```bash
PYTHONPATH=$PWD/src /remote-home/zty/conda/LM_zty/repetition/bin/python \
  scripts/build_pseudo_safety_direction.py \
  --model /remote-home/model/llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct \
  --dataset-keys jbb-llama-ofa,jbb-llama-pair,jbb-llama-drattack \
  --metric alr \
  --output-dir outputs/pseudo_safety_direction \
  --prefix mixed_alr_last_layer
```

输出：

- `mixed_alr_last_layer_direction.pt`：Ridge 方向、intercept、alpha 和元数据。
- `mixed_alr_last_layer_scores.jsonl`：每条 q1 的 ASR、ALR、split、pseudo_score。
- `mixed_alr_last_layer_direction_report.json`：方向在训练/测试上的相关性。

### 2. 分攻击方案构造 pairwise 伪标签

```bash
PYTHONPATH=$PWD/src /remote-home/zty/conda/LM_zty/repetition/bin/python \
  scripts/build_pseudo_safety_pairs.py \
  --scores outputs/pseudo_safety_direction/mixed_alr_last_layer_scores.jsonl \
  --dataset-key jbb-llama-ofa \
  --output-dir outputs/pseudo_safety_direction/pairs/mixed_alr_last_layer
```

同理分别跑 `jbb-llama-pair` 和 `jbb-llama-drattack`。

输出：

- `train_pairs.json`
- `test_pairs.json`
- `pairs_summary.json`

### 3. 训练新的 CompSP 式分类器

```bash
PYTHONPATH=$PWD/src /remote-home/zty/conda/LM_zty/repetition/bin/python \
  scripts/train_pseudo_direction_compsp.py \
  --train-file outputs/pseudo_safety_direction/pairs/mixed_alr_last_layer/jbb-llama-ofa/train_pairs.json \
  --test-file outputs/pseudo_safety_direction/pairs/mixed_alr_last_layer/jbb-llama-ofa/test_pairs.json \
  --output-dir outputs/pseudo_safety_direction/models/mixed_alr_last_layer/jbb-llama-ofa
```

输出：

- `lora/`：新训练 LoRA。
- `tokenizer/`：配套 tokenizer。
- `train_summary.json`：初始评估、训练指标、最终测试指标。

### 4. 一键运行

```bash
CUDA_VISIBLE_DEVICES=1 nohup scripts/run_pseudo_safety_learnability.sh \
  > logs/pseudo_safety_learnability_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

烟测：

```bash
SMOKE=1 CUDA_VISIBLE_DEVICES=1 scripts/run_pseudo_safety_learnability.sh
```

ASR 辅助实验使用可续跑启动器：

```bash
CUDA_VISIBLE_DEVICES=1 scripts/launch_pseudo_safety_asr.sh
```

启动器设置 `METRIC=asr` 和 `SKIP_EXISTING=1`。方向分数、pair 数据或某个攻击方案的
`train_summary.json` 已存在时，会逐阶段跳过，避免中断后重复长时间计算。

## 第一轮实验顺序

1. 混合 OFA+PAIR+DrAttack 训练 ALR-Ridge 主方向。
2. 用同一个方向分别为 OFA、PAIR、DrAttack 构造伪标签。
3. 三个攻击方案分别训练三个新 CompSP 模型。
4. 报告每个模型在对应测试题上的 pairwise accuracy、F1、AUC、标签分布。
5. 再跑 ASR-Ridge 辅助方向。
6. 如果三组单独训练结果稳定，再训练混合数据模型，测试跨攻击方案泛化。

## 主结论标准

可以支持的结论：

> CompSP 式 pairwise 文本代理能够学习由白盒最后层表示定义的安全相关一维结构。

暂时不能直接支持的结论：

- 学到的就是学界定义的真实安全向量。
- 旧 CompSP ASR 模型已经学到了这个方向。
- Qwen/4omini 上的 CompSP 可以解释 Llama 的隐藏层结构。

## 当前限制

- 第一版只使用最后一层最后 token。
- Ridge 方向是一维；多维安全结构需要后续扩展到 PLS/PCA/子空间。
- ALR 是主监督，因为它更接近拒绝/长度行为；ASR 仍需作为最终攻击成功指标报告。
## 2026-07-10 阶段性落档：ALR 非边界子集与 DrAttack 进度

### 远程运行状态

- 2026-07-10 09:40（北京时间）检查时，`jbb-llama-drattack` 伪安全方向 CompSP 训练仍在运行。
- 最新 tqdm 约为 `12372/14108`，进度约 `87.7%`。
- tqdm 训练剩余约 `3h`；由于 DrAttack 测试集有 `43,367` 个 pair，最终 evaluation/save 预计另需 `1.3~1.6h`。
- 保守预计完整结束时间为北京时间 `2026-07-10 14:00` 左右。
- DrAttack 未生成 `train_summary.json` 前，不启动 ASR 全流程，避免抢占 GPU。

### ALR 非边界子集检查

为排除模型只学习原始 ALR 的 `0 / middle / 1` 粗桶而形成 reward hacking，新增离线评估脚本：

```text
scripts/evaluate_pseudo_direction_compsp_subsets.py
scripts/run_pseudo_subset_eval.sh
```

筛选条件：pair 中两条指令的原始 ALR 都严格位于 `(0,1)`。

结果文件：

```text
outputs/pseudo_safety_direction/subset_eval/mixed_alr_last_layer/jbb-llama-ofa_alr_middle_eval.json
outputs/pseudo_safety_direction/subset_eval/mixed_alr_last_layer/jbb-llama-pair_alr_middle_eval.json
```

| 数据集 | 全测试 accuracy | 两端 ALR 都在 (0,1) 的 n | 两端 ALR 都在 (0,1) accuracy | 同 ALR 桶 accuracy | 不同 ALR 桶 accuracy |
|---|---:|---:|---:|---:|---:|
| OFA | 0.7237 | 3,050 | 0.6443 | 0.5797 | 0.8239 |
| PAIR | 0.7602 | 778 | 0.6735 | 0.6198 | 0.8381 |

阶段性解释：

1. 完整测试准确率中确实有一部分来自 ALR 粗桶差异，尤其是边界值 `0` 和 `1` 形成的容易 pair。
2. 但在两端 ALR 都非边界的子集上，OFA/PAIR 仍显著高于随机，说明当前正结果不能完全由朴素二分类/三分类桶捷径解释。
3. 同 ALR 桶子集 accuracy 更低，提示后续需要补充随机标签、打乱投影、随机方向 baseline 和按问题宏平均 accuracy。
4. DrAttack 的同类评估需等待当前训练完成后补跑，预计单次评估约 `1.3~1.6h`。
