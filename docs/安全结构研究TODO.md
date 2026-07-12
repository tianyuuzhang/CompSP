# 安全结构研究 TODO

更新时间：2026-07-13

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

## 2026-07-08 补充：伪安全结构可学习性实验

训练集固定为 `0~79` 中排除旧论文 20 个测试题后的 60 个问题；旧 CompSP LoRA 和旧 BTL 分数与目标模型、攻击方案、训练指标绑定，不能复用。本实验的新训练目标是白盒最后层 Ridge 伪安全方向投影的 pairwise 序关系，主证据是新训练 CompSP 在留出测试题上的 pairwise accuracy，BTL 只是可选后续分析。详见 `docs/pseudo_safety_learnability.md`。

## 2026-07-11 新增：回答嗅探与替代 ASR

- [x] 单/少量回答手工特征、TF-IDF 与学习曲线。
- [x] `仅q1/仅回答/q1+回答` 三视图对照。
- [x] 同攻击方案、同问题内打乱回答归属负对照。
- [x] 单回答 Llama 最后层 embedding 初验；结果未超过黑盒 TF-IDF。
- [x] 实现 Jeffreys ASR、判定 hazard、首票/延迟 unsafe、三票全 safe 和错误票率。
- [x] 验证 Llama 三攻击替代指标的相关性、分辨率和重建一致率。
- [x] 扩展到 Qwen、4omini、DeepSeek 的 OFA 跨模型矩阵，按目标模型自身标签报告；HarmBench 仍待补。
- [x] 训练单模型嗅探器并做 OFA/PAIR/DrAttack 跨攻击迁移矩阵。
- [x] 做跨模型迁移矩阵；禁止拿源模型 ASR 当目标模型标签。
- [x] 增加 OFA 跨模型 `q1-only` baseline，用于检验回答文本是否提供输入之外的增量信号。
- [x] 严格前缀/后缀拆分：只用前 k 条回答文本预测未使用后缀 ASR。
- [x] 对跨模型矩阵增加去拒绝词、去长度、去列表/格式特征后的残差版本。
- [ ] 用新采样或不同温度回答复验少回答文本预测后续 ASR，避免历史 20 次内部拆分的偶然性。
- [ ] 抽样重新进行完整三票判定，比较 any-unsafe、majority-unsafe 与概率加权标签。
- [ ] 增加 Brier、ECE、可靠性曲线及 Beta/二项置信区间覆盖。
- [x] 设计序贯采样，在固定预算下评估低 ASR 区分能力。
- [x] 明确报告：现有经验 ASR 分辨率为 0.05，0.01/0.001 只能作为潜在风险预测或增采目标。

## 2026-07-12 修正：结构探索与结构利用口径

- [ ] 黑盒特征只使用攻击者可获得的文本权限：q1、模型回答、以及由文本派生的统计特征；白盒 hidden state 只用于机制解释和参照。
- [ ] 回答分析不局限 Llama；只要标签来自同一目标模型，Qwen、4omini、DeepSeek、HarmBench 都可纳入回答内容与 ASR/ALR 相关性分析。
- [ ] 优先围绕 ASR 建立利用场景：用单条或少量回答定位指令在风险结构中的相对位置，并探索能否为低 ASR 区间提供优化信号。
- [ ] 探索阶段允许使用 ALR、拒绝形式、替代 ASR、伪方向投影等任意有效目标；结论不预设，按实验结果调整主假设。
- [ ] 对 ASR 定义做敏感性分析：any-unsafe、判定 hazard、加权 ASR、未来可抽样验证的多数 unsafe；旧数据中因提前停止导致无法完全离线重建三票 unsafe。
- [ ] 输入、输出、输入+输出三种视图都保留；重点报告输出相对 q1-only 的增量，避免把输入攻击模板误解释为回答泄露。
- [ ] 在数据混合之外，保留按攻击方案、按模型、按数据集的分层结果；迁移性作为结构稳定性的辅助证据。

## 2026-07-12 新增：跨模型残差矩阵后的判断

- [x] JBB-OFA 四模型残差矩阵完成；ASR 和 hazard 加权 ASR 在扣除长度/拒绝/格式后仍有文本增量，ALR 基本没有。
- [x] 针对 ASR 残差本身做高权重词组审计，确认剩余信号是否来自具体危险主题、替代建议、语气姿态或攻击模板残片。
- [x] 在 HarmBench OFA 上复跑同样残差矩阵，检查数据集迁移是否保持。
- [x] 做按目标模型分层的校准/误差分析：确认哪些目标模型的 ASR 残差信号最强，是否与基础 ASR 方差有关。
- [x] 对 HarmBench 中 Llama 目标的异常格子做单独审计，确认 qwen/deepseek 训练源在 Llama ASR 上的长度格式基线偏低是否来自题目数量小或分布差异。
- [ ] 若后续资源允许，扩大 HarmBench 测试问题或重新采样，验证 Llama/Qwen 目标上的 ASR 残差增益是否稳定。

## 2026-07-12 新增：低 ASR 分辨率限制

- [x] 统计经验 ASR 档位、低 ASR 比例和 Jeffreys 后验区间；当前 `totQA=20` 无法直接区分 `0.01` 与 `0.001`。
- [ ] 设计序贯增采策略：先用少量回答文本/残差信号筛选候选，再对低 ASR 候选追加采样以缩小置信区间。
- [ ] 若目标是低 ASR 精细定位，考虑跨同题 prompt 族的层级模型，而不是逐指令独立估计概率。

## 2026-07-12 新增：序贯 ASR 增采模拟

- [x] 离线模拟前缀/后缀拆分：只用前 `k=1,2,4` 条回答选择 top 20% 候选，再用未使用后缀 ASR 评估。
- [x] Llama 三攻击首轮完成：随机后续 ASR 约 `0.636`；`k=1/2/4` 的 hazard-weighted 策略分别约 `0.770/0.802/0.824`。
- [x] 增加跨模型版本：JBB-OFA 的 Qwen、4omini、DeepSeek 各自按自身 ASR 标签评估，检查少量回答增采信号是否跨目标模型稳定。
- [x] 增加 HarmBench 版本；由于 HarmBench 测试题少、部分模型 ASR/ALR 方差不足，需要同时报告题目数和标签方差。
- [ ] 将当前“前缀 unsafe 几乎决定选择”的策略与文本残差嗅探器结合，重点寻找前缀零 unsafe 但后缀高风险的隐蔽候选。
- [ ] 设计真正预算模拟：给定总访问预算，比较随机均匀采样、经验 ASR 增采、ALR 辅助增采和文本模型增采对高风险指令发现率的影响。

## 2026-07-12 新增：前缀文本残差增采

- [x] 实现 `simulate_text_residual_sequential_asr.py`：训练题拟合前缀回答到后缀 ASR，测试题上按预测分数选择 top 20%。
- [x] Llama 三攻击轻量首轮完成：文本模型有 ASR lift，但没有解决前缀零 unsafe 隐蔽高风险召回低的问题。
- [x] 加入 prefix-zero 保留名额混合策略，确认隐蔽高风险召回可以从接近 0 提升到 `0.38~0.69`，代价是平均 ASR lift 明显下降。
- [x] 对前缀零 unsafe 样本单独训练/评估二阶段模型，目标从“整体 ASR lift”改成“隐蔽高风险召回”，并报告 AP、AUC、top precision/recall。
- [ ] 给 prefix-zero 二阶段模型补多 seed 置信区间，并按 OFA/PAIR/DrAttack 分别报告。
- [x] 在 prefix-zero 子集中审计高权重文本特征，确认模型是否依赖具体危险内容、回答姿态、长度格式残余或模板痕迹。
- [ ] 对 prefix-zero 高权重词项做同题、同攻击方案内的主题控制审计，区分真实回答姿态信号与题目/模板残留。
- [ ] 对 prefix-zero 二阶段模型增加 q1-only、response-only、q1+response 三视图对照，明确回答文本相对输入攻击文本的增量。
- [ ] 对文本残差脚本增加长跑配置：多 seed、word+char n-gram、分攻击方案报告；预计三攻击全量多 seed 需要 30 分钟到数小时，需写日志和进度。
- [ ] 将序贯模拟改成真实预算曲线：初始每条 1 次回答，然后只对 top 候选追加采样，比较固定预算下发现高风险指令的效率。
- [ ] 把预算策略区分成两类目标：攻击优化优先平均 future ASR，安全审计优先 prefix-zero 隐蔽高风险召回。
