# 第一次 Llama 白盒对齐实验

运行脚本：

```bash
cd /remote-home/zty/tidy_CompSP
CUDA_VISIBLE_DEVICES=1 nohup scripts/run_whitebox_llama_alignment.sh > logs/whitebox_llama_alignment_20260706_034612.log 2>&1 &
```

环境：

- Python：`/remote-home/zty/conda/LM_zty/repetition/bin/python`
- 模型：`/remote-home/model/llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct`
- GPU：A6000 第 1 号卡

生成文件：

- `outputs/whitebox/llama31_instruct_refusal_direction_n80.pt`
- `outputs/whitebox/jbb-llama-pair_whitebox_scores.jsonl`
- `outputs/whitebox/jbb-llama-pair_alignment_alr.json`
- `outputs/whitebox/jbb-llama-pair_alignment_asr.json`
- `outputs/whitebox/jbb-llama-drattack_whitebox_scores.jsonl`
- `outputs/whitebox/jbb-llama-drattack_alignment_alr.json`
- `outputs/whitebox/jbb-llama-drattack_alignment_asr.json`

## 数据对齐

这次实验修正了之前的模型-标签错配：

- `jbb-llama-pair` 对应 `data/saverk/save_jbb2100_pair_llama`
- `jbb-llama-drattack` 对应 `data/saverk/save_jbb4100_dr_ins_llama`
- `item_index` 用来匹配 `ranking_scores.json` 里的 `item_id`；Pair 数据里的 `round` 不可靠。

## 方向

方向构造方式：

```text
direction[layer] = mean(last_token_hidden(harmful JBB prompts)) - mean(last_token_hidden(benign prompts))
```

Anchors：

- 80 个 harmful prompt，来自 `/remote-home/zty/data/jailbreak_questions.json`
- 80 个 benign prompt，来自 `/remote-home/zty/data/data_benign-behaviors.csv`

构造时分离最好的一层：layer 32。

## 结果摘要

### JBB Llama Pair, ALR

- 和 BTL 匹配的行数：2000
- 可比较 pair 数：20663
- CompSP 对 ALR 的准确率：0.806
- 校准后的白盒方向对 ALR 的最好准确率：layer 28、sign -1、0.850
- CompSP / 白盒方向最好一致率：layer 24、sign -1、0.743
- 未校准情况下的最佳 wrong-agreement：layer 28、sign +1、0.777

解释：

白盒方向在符号校准后，对 ALR 有很强的预测力。sign +1 下出现的高 wrong-agreement，主要是符号反了导致的：sign +1 和 ALR 是反相关的，所以它自然会在 CompSP 出错时“看起来同错”。这不能当作“同一种错误结构”的证据。

### JBB Llama Pair, ASR

- 可比较 pair 数：36139
- CompSP 对 ASR 的准确率：0.729
- 校准后的白盒方向对 ASR 的最好准确率：layer 32、sign -1、0.640
- CompSP / 白盒方向最好一致率：layer 23、sign -1、0.659
- 最佳 wrong-agreement：layer 5、sign +1、0.554

解释：

ASR 比较噪声大，也没有那么直接地跟简单的 harmful-minus-benign 方向对齐。这和论文里“ALR 比 ASR 更客观地反映拒绝行为”是一致的。

### JBB Llama DrAttack, ALR

- 和 BTL 匹配的行数：2000
- 可比较 pair 数：30557
- CompSP 对 ALR 的准确率：0.629
- 校准后的白盒方向对 ALR 的最好准确率：layer 20、sign -1、0.853
- CompSP / 白盒方向最好一致率：layer 20、sign -1、0.584
- 未校准情况下的最佳 wrong-agreement：layer 20、sign +1、0.862

解释：

这个简单白盒方向对 DrAttack 的 ALR 预测非常强，而 CompSP 排序反而弱一些。同样，未校准的高 wrong-agreement 主要还是强 ALR 方向的反向符号。

### JBB Llama DrAttack, ASR

- 可比较 pair 数：36629
- CompSP 对 ASR 的准确率：0.604
- 校准后的白盒方向对 ASR 的最好准确率：layer 21、sign -1、0.743
- CompSP / 白盒方向最好一致率：layer 32、sign -1、0.584
- 最佳 wrong-agreement：layer 21、sign +1、0.718

解释：

这个白盒方向在 DrAttack 场景下比 CompSP 更能预测 ASR，但这还不能说明 CompSP 和白盒方向共享同一种错误。

## 当前结论

这次初跑支持：

1. Llama 的 ALR/ASR 行为确实有很强的低维白盒信号。
2. 这个信号在 ALR 上尤其强，说明 ALR 更像拒绝/长度行为的直接代理。
3. CompSP 排序和白盒方向并非完全无关，但在做了符号校准之后，“错得一样”这一假设还没有被支持。

sign +1 下的高 wrong-agreement 更像是一个诊断性伪影，除非这个符号有理论依据。更公平的做法是：先在训练 anchor 上校准白盒方向的符号，再在留出的测试问题上看 wrong-agreement。

## 下一步实验

1. 先在训练问题上做符号校准，再在留出问题上评估 wrong-agreement。
2. 按照 “LLMs Encode Harmfulness and Refusal Separately” 的思路，分开构造有害性方向和拒绝方向。
3. 测试多维方向：对 harmful/benign activation difference 做 PCA/SVD，或者复现 orthogonal safety directions。
4. 给 wrong-agreement 加 shuffle / permutation baseline。
5. 以后如果能找到 OFA 对应的 BTL 排名，再把 `jbb-llama-ofa` 加进来一起比。
