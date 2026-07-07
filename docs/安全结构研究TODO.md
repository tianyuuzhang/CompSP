# 安全结构研究 TODO

更新时间：2026-07-07

## 最高优先级

- [ ] 修正白盒误差对齐评估：训练题校准 layer/sign，测试题固定评估。
- [ ] 给 `analyze_whitebox_compsp_alignment.py` 增加同题内 permutation baseline 和 bootstrap 置信区间。
- [ ] 单独输出 ALR 与 ASR 的结果表，禁止在脚本里混用指标。
- [ ] 生成错排案例表：高 ASR 低 CompSP、低 ASR 高 CompSP，各保留 q1、ASR、ALR、BTL、whitebox 分数。
- [ ] 在文档中明确：未校准 sign 的 wrong-agreement 只能作为诊断，不作为主证据。

## 白盒方向实验

- [ ] 复查第一次 Llama 白盒方向输出，确认 layer 32 被选为 anchor separation 最好层是否合理。
- [ ] 增加 train/test question split，默认沿用论文 20 个测试题之外的问题做校准。
- [ ] 构造 harmfulness direction 与 refusal direction 两套 anchor。
- [ ] 测试 last token、mean token、assistant start token 三种池化位置。
- [ ] 记录每层方向范数、harmful/benign separation、方向间 cosine。
- [ ] 增加随机同范数方向 baseline。

## CompSP 对齐实验

- [ ] Pair：`jbb-llama-pair` + `data/saverk/save_jbb2100_pair_llama`。
- [ ] DrAttack：`jbb-llama-drattack` + `data/saverk/save_jbb4100_dr_ins_llama`。
- [ ] 检查是否存在 Llama OFA 的 BTL 排名；若没有，先不要把 OFA 放入 CompSP-白盒主表。
- [ ] 所有匹配使用 JSON 列表下标 `item_index`，不要用不可靠的 `round`。
- [ ] 对每个问题分别计算指标，再宏平均和微平均都报告。

## 多维结构实验

- [ ] 对 harmful/benign activation shift 做 PCA/SVD。
- [ ] 评估 top-k 子空间投影范数与 ASR/ALR/BTL 的相关。
- [ ] 测试 top-k = 1, 2, 4, 8, 16，并在开发集选 k。
- [ ] 参考 Pan et al.，解释次级方向是否对应 role-play、hypothetical、translation、obfuscation 等攻击风格。
- [ ] 如果 SOM/refusal manifold 方案成本可控，再考虑复现 Piras et al. 的多方向方法。

## 数据与工程

- [ ] 给 `outputs/whitebox` 写中文 README，说明哪些是可复现结果、哪些是临时文件。
- [ ] 把白盒实验配置整理成 `configs/whitebox_alignment.yaml` 或等价 shell 配置。
- [ ] 给所有新脚本加中文 docstring/注释，说明输入、输出、核心逻辑。
- [ ] 保证大数据从 `data/asr_datasets.json` 或解包归档读取，不硬编码旧路径。
- [ ] 每次长实验写入 `logs/`，结果写入 `outputs/whitebox/`。

## 文献与写作

- [ ] 阅读 Arditi 单方向拒绝论文，整理复现细节。
- [ ] 阅读 Zhao harmfulness/refusal 分离论文，设计双方向 anchor。
- [ ] 阅读 Pan 多维安全方向论文，确定 PCA/SVD baseline。
- [ ] 阅读 Wang 多语言拒绝方向论文，判断跨主题/跨语言泛化是否能作为类比。
- [ ] 阅读 Joad/Piras 多方向 refusal 文献，决定是否需要从单方向升级到 manifold。
- [ ] 在 `docs/安全结构研究讨论.md` 中持续记录每次实验后改变了什么判断。

## 暂缓事项

- [ ] 不急于做跨模型强结论；Qwen/4omini 只做对照。
- [ ] 不急于写完整论文；先把核心证据链固定。
- [ ] 不急于做 SAE，除非单方向和 PCA 都无法解释误差。
