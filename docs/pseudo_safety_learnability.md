# 伪安全结构可学习性实验

更新时间：2026-07-11

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

## 2026-07-11 落档：DrAttack ALR 完整结果

DrAttack 训练与最终测试已经结束：

| 指标 | 结果 |
|---|---:|
| 训练 pair | 112,871 |
| 测试 pair | 43,367 |
| 训练耗时 | 98,787 秒（27.44 小时） |
| 测试 accuracy | 0.4968 |
| 测试 macro-F1 | 0.4575 |
| 测试 AUC | 0.4967 |

该结果在当前设置下等同随机，说明“ALR 监督得到的混合白盒一维投影可被 CompSP
跨问题学习”只在 OFA 和 PAIR 上得到支持，不能推广到 DrAttack。DrAttack 的文本更长、
更模板化且大量输入达到 512-token 截断上限，可能使 pairwise 分类器无法看到决定投影
序关系的有效部分；这只是待检验解释，不能作为既定原因。

DrAttack 非边界 ALR 子集评估于 2026-07-11 在 GPU 0 启动，预计 1.3~1.6 小时完成。
ASR 辅助方向实验同时在 GPU 1 启动，PID 为 `3885687`，初始日志为
`logs/pseudo_safety_asr_20260711_013330.log`。方向提取约 20 分钟完成，测试题 Pearson
为 `0.8067`、Spearman 为 `0.7866`。OFA 构造出 100,010 个训练 pair 和 40,757 个
测试 pair，现已进入 OFA 模型阶段。按 ALR 三组实际耗时重新估计，ASR 全流程需要
约 45~55 小时；此前 25~35 小时估计偏乐观。产生稳定训练 tqdm 后继续收紧 ETA。

DrAttack 非边界子集评估最终结果：

| 子集 | n | accuracy | AUC |
|---|---:|---:|---:|
| 全测试 | 43,367 | 0.4968 | 0.4967 |
| 两端 ALR 均在 `(0,1)` | 2,722 | 0.4871 | 0.5310 |
| 同 ALR 粗桶 | 16,594 | 0.4880 | 0.4793 |
| 不同 ALR 粗桶 | 26,773 | 0.5023 | 0.5076 |

DrAttack 在非边界、中间区间以及同/不同粗桶中都接近随机，因此其负结果不是由
ALR=0/1 边界样本掩盖，也不是简单的粗桶分布问题。后续需要优先检查 512-token 截断、
DrAttack 指令有效信息位置和 LoRA 训练稳定性，而不是继续解释为 reward hacking。

## 2026-07-12 落档：ASR-PAIR 完成，ASR-DrAttack 启动

ASR-Ridge 方向下，`jbb-llama-pair` 的新 CompSP 训练已经完成：

| 数据集 | 训练 pair | 测试 pair | 训练耗时 | final accuracy | macro-F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| jbb-llama-pair | 108,665 | 44,204 | 35,339s | 0.6485 | 0.6484 | 0.7086 |

解释：ASR 伪方向在 PAIR 上可以被 CompSP 学到，但明显弱于 ALR 方向上的 PAIR
结果（accuracy 约 0.7602）。这符合当前判断：ASR 更接近最终攻击成功指标，受采样、
判定器和具体危险内容影响更大；ALR/拒绝结构更容易被文本和 pairwise 模型捕获。

`jbb-llama-drattack` 已自动启动。pair 构造结果：

| 数据集 | 训练 pair | 测试 pair | 训练问题 | 测试问题 |
|---|---:|---:|---:|---:|
| jbb-llama-drattack | 115,913 | 48,417 | 60 | 20 |

初始 ETA：参考 ALR-DrAttack 实测 27.44 小时训练和约 1.3 小时最终评估，ASR-DrAttack
预计约 28~32 小时。等训练 tqdm 正式出现后，以日志 ETA 为准。

## 2026-07-12 续档：ASR-DrAttack 进入训练阶段

`jbb-llama-drattack` 已结束初始 evaluation 并进入 LoRA 训练。日志：
`logs/pseudo_safety_asr_20260711_013330.log`。

2026-07-12 00:40（北京时间）检查时，训练 tqdm 约为 `6277/14490`，进度约 `43%`，
速度约 `6.1s/step`，剩余时间约 `14h`。训练完成后还会进行最终 evaluation/save，
按测试集 `48,417` 个 pair 估计另需约 `1.3~1.7h`。保守预计完整结束还需 `15~16h`。

当前已有 ASR 结果：

| 数据集 | final accuracy | macro-F1 | AUC |
|---|---:|---:|---:|
| jbb-llama-ofa | 0.6103 | 0.6103 | 0.6580 |
| jbb-llama-pair | 0.6485 | 0.6484 | 0.7086 |
| jbb-llama-drattack | 训练中 | 训练中 | 训练中 |

阶段性判断：ASR 伪方向比 ALR 方向更难被 CompSP 拟合，但 OFA/PAIR 均显著高于随机。
DrAttack 需等待最终结果；鉴于 ALR-DrAttack 曾接近随机，不能预设 ASR-DrAttack 会成功。

## 2026-07-13 落档：ASR 三攻击完整结果

ASR 伪安全方向三组 CompSP 训练均已完成。训练目标为白盒 Llama 最后一层 Ridge
ASR 伪方向投影诱导的 pairwise 序关系；测试集仍为旧论文 20 个测试题。

| 攻击方案 | 初始 accuracy | 初始 AUC | 最终 accuracy | macro-F1 | AUC | 训练耗时 |
|---|---:|---:|---:|---:|---:|---:|
| OFA | 0.4975 | 0.4999 | 0.6103 | 0.6103 | 0.6580 | 9.54 h |
| PAIR | 0.4971 | 0.5047 | 0.6485 | 0.6484 | 0.7086 | 9.82 h |
| DrAttack | 0.4978 | 0.4896 | 0.5656 | 0.5640 | 0.5923 | 27.62 h |

对应文件：

- `outputs/pseudo_safety_direction/models/mixed_asr_last_layer/jbb-llama-ofa/train_summary.json`
- `outputs/pseudo_safety_direction/models/mixed_asr_last_layer/jbb-llama-pair/train_summary.json`
- `outputs/pseudo_safety_direction/models/mixed_asr_last_layer/jbb-llama-drattack/train_summary.json`

解释：ASR 伪方向在 OFA 和 PAIR 上可被 CompSP 学到，PAIR 最明显；DrAttack 也高于
随机，但幅度较弱。这个结果比 ALR-DrAttack 的近随机结果更积极，说明 ASR 方向上的
可学习结构并不完全依赖 OFA/PAIR，不过 DrAttack 的泛化强度明显不足，不能把三攻击
混成同一强结论。

从研究叙事看，这组结果支持“安全结构可学习”的弱到中等证据链：黑盒文本代理可以在
留出问题上拟合白盒伪安全方向诱导的序关系，但攻击方案差异很大。后续若要把它提升为
主证据，需要补充：

- 多 seed 或至少一次轻量复验，确认 DrAttack 弱正结果稳定；
- frozen embedding / hidden state 对 response 信号的对齐实验；
- BTL 或排序层面的可选分析，但不要把 BTL 当作结构可学习性的必要条件。
