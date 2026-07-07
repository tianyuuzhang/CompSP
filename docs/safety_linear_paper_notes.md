# 安全线性结构论文笔记

PDF 文件存放在 `/remote-home/zty/tidy_CompSP/参考文献`。

## 1. Arditi et al. 2024 - Refusal in Language Models Is Mediated by a Single Direction

PDF：`2024_Arditi_Refusal_Single_Direction.pdf`

链接：https://arxiv.org/abs/2406.11717

核心结论：很多聊天模型的拒绝行为可以被一个一维残差流方向所中介。去掉这个方向会降低对有害提示词的拒绝，加入这个方向会让无害提示词也表现出拒绝。

方法：

- 使用有害/无害对照数据。
- 提取残差流激活。
- 找出拒绝方向。
- 通过加上或消去该方向做因果干预。

对 CompSP 的意义：

- 这是最简单的白盒对象，可以直接拿来和 CompSP 分数比较。
- 我们的第一版复现应该在 Llama-3.1-8B-Instruct 上构造一个拒绝方向，然后看 CompSP 排序误差是否与它对齐。

可复用指标：

- 每层上 `q1` 到拒绝方向的投影。
- 两两标签：`proj(q1_i) > proj(q1_j)`。

## 2. Zou et al. 2023 - Representation Engineering: A Top-Down Approach to AI Transparency

PDF：`2023_Zou_Representation_Engineering.pdf`

链接：https://arxiv.org/abs/2310.01405

核心结论：诚实、无害、真实、趋利等高层概念和行为，往往可以通过群体级表示被监控或操控。

方法：

- 构造对照概念数据集。
- 提取隐藏状态。
- 做表示读取和表示控制。

对 CompSP 的意义：

- 为把 CompSP 学到的排序看成黑盒版表示方向提供了概念框架。
- 也支持先用简单的对照 hidden-state 方向作为第一版 baseline。

可复用指标：

- `q1` 上的线性 probe 或概念方向分数。

## 3. Xu et al. 2024 - Uncovering Safety Risks of Large Language Models through Concept Activation Vector

PDF：`2024_Xu_SCAV_Safety_Concept_Activation_Vector.pdf`

链接：https://arxiv.org/abs/2404.12038

核心结论：安全概念激活向量（SCAV）可以解释安全机制，也可以指导攻击者沿着安全相关方向移动 prompt 或 embedding。

方法：

- 从激活中构造安全概念向量。
- 用它们指导 prompt 级和 embedding 级攻击。

对 CompSP 的意义：

- 跟“把潜在安全分数当成攻击优化信号”这件事非常接近。
- 它提示我们，白盒方向分数未必和 ASR 完全一致，但依然可能指导优化，这正是我们想分析的差异。

可复用指标：

- 每个 `q1` 的 SCAV 投影分数。
- SCAV 分数和 CompSP BTL 分数的相关性。

## 4. Lin et al. 2024 - Towards Understanding Jailbreak Attacks in LLMs: A Representation Space Analysis

PDF：`2024_Lin_Jailbreak_Representation_Space_Analysis.pdf`

链接：https://arxiv.org/abs/2406.10794

核心结论：成功的越狱会把有害 prompt 的表示推向无害/可接受 prompt 的表示区域。Harmful 和 harmless prompt 在对齐后的 Llama 表示里是可分的，低维成分解释了很多这种分离。

方法：

- 比较有害、无害、越狱 prompt 的隐藏表示。
- 使用 PCA 和表示空间目标。

对 CompSP 的意义：

- 直接支持一种输入侧视角：prompt 自己在生成前就可能沿着安全表示空间移动。
- 我们的 OFA `q1` 可以用来打分，衡量它离有害/无害区域有多远。

可复用指标：

- 接受方向投影。
- 到有害/无害中心的距离。

## 5. Cao et al. 2024 - SCANS: Safety-Conscious Activation Steering

PDF：`2024_Cao_SCANS_Safety_Conscious_Activation_Steering.pdf`

链接：https://arxiv.org/abs/2408.11491

核心结论：拒绝 steering 向量可以从有害-vs-无害的 hidden-state 对照中提取出来。中间层尤其关键；条件式 steering 可以在保留防护的同时减少过度拒答。

方法：

- 用 harmful 和 benign anchor 计算层级拒绝向量。
- 用词表投影找安全关键层。
- 根据输入相似度做条件 steering。

对 CompSP 的意义：

- 这是最适合第一版低成本复现的路线。
- 层级投影很适合解释为什么 CompSP 会把某些 prompt 排高/排低。

可复用指标：

- 每层的拒绝向量投影。
- 最优层和中间层平均投影。

## 6. Lee et al. 2024 - Programming Refusal with Conditional Activation Steering

PDF：`2024_Lee_CAST_Conditional_Activation_Steering.pdf`

链接：https://arxiv.org/abs/2409.05907

核心结论：activation steering 可以做成条件式的，只在 prompt 匹配某个条件向量时才施加拒绝向量。这可以实现选择性拒绝规则。

方法：

- 学习或定义条件向量。
- 用相似度决定是否施加拒绝向量。

对 CompSP 的意义：

- CompSP 学到的可能不只是一个全局拒绝方向，而是一个依赖 prompt 结构的条件排序。
- 这篇论文提示我们要按攻击风格或 prompt 聚类去看白盒/CompSP 的对齐。

可复用指标：

- 和条件向量的相似度。
- 按攻击风格或 prompt 簇做条件一致性分析。

## 7. O'Brien et al. 2024 - Steering Language Model Refusal with Sparse Autoencoders

PDF：`2024_OBrien_SAE_Refusal_Steering.pdf`

链接：https://arxiv.org/abs/2411.11296

核心结论：稀疏自编码器特征可以介导拒绝；放大这些特征会增强拒绝鲁棒性，但可能伤害通用能力。

方法：

- 训练或复用 SAE 特征。
- 找出介导拒绝的特征。
- 在推理时 clamp 或 amplify 特征。

对 CompSP 的意义：

- 说明拒绝可能不是单一方向，而是分布在一组特征里，与一般能力纠缠。
- 如果单方向对齐不强，SAE 或多特征解释就是后续很自然的扩展。

可复用指标：

- 不是第一版优先项，除非已有 SAE 资源。
- 更适合作为概念控制：单方向可能不够。

## 8. Gao et al. 2024 - Shaping the Safety Boundaries

PDF：`2024_Gao_Safety_Boundaries_ABD.pdf`

链接：https://arxiv.org/abs/2412.17034

核心结论：越狱会把有害激活推过安全边界；低层和中层很关键。Activation Boundary Defense 会把激活约束在这个边界内。

方法：

- 分析有害和越狱 prompt 的激活分布。
- 定义安全边界。
- 做层级激活约束。

对 CompSP 的意义：

- 这和“黑盒交互能揭示潜在结构”的故事非常一致。
- 它提示我们不只看一维投影，还要看 boundary 距离。

可复用指标：

- 到有害/无害区域的马氏距离或中心距离。
- 按层的边界越界分数。

## 9. Pan et al. 2025 - The Hidden Dimensions of LLM Alignment

PDF：`2025_Pan_Hidden_Dimensions_Orthogonal_Safety_Directions.pdf`

链接：https://arxiv.org/abs/2502.09674

核心结论：安全行为不是只有一个方向；除了主拒绝方向外，正交的次级方向还编码了角色扮演、假设叙述等可解释特征。

方法：

- 在 Llama 3 8B 的安全微调过程中研究表示变化。
- 把变化分解成正交方向。
- 分析主次方向之间的关系。

对 CompSP 的意义：

- OFA prompt 在形式和叙事结构上差异很大，所以次级方向可能比单纯 ASR 更能解释 CompSP 的“错排”。
- 后续应该从单方向推进到小型子空间打分。

可复用指标：

- 投影到 top-k 安全方向。
- 比较 CompSP 误差和次级方向的关系。

## 10. Zhao et al. 2025 - LLMs Encode Harmfulness and Refusal Separately

PDF：`2025_Zhao_Harmfulness_Refusal_Separately.pdf`

链接：https://arxiv.org/abs/2507.11878

核心结论：有害性和拒绝是两个不同的潜在概念。越狱可能只是减少了拒绝信号，而没有消除模型内部的有害性表示。

方法：

- 找出有害性方向和拒绝方向。
- 比较指令处理前后的 token 位置。
- 用 steering 证明两者在因果上可分。

对 CompSP 的意义：

- 这是解释 CompSP 误差最重要的一篇。
- 一个 prompt 可以是“高有害性但低拒绝”，也可以是“低有害性但高拒绝”。经验 ASR 依赖两者，而 CompSP 可能更靠近其中一个维度。

可复用指标：

- 指令末尾 token 的有害性投影。
- 指令后/生成起始 token 的拒绝投影。
- 二维比较：CompSP 分数 vs 有害性/拒绝坐标。

## 优先阅读顺序

1. Arditi 单方向拒绝论文。
2. Zhao 有害性/拒绝分离论文。
3. Pan 多维安全方向论文。
4. Lin 越狱表示空间分析论文。
5. SCANS，作为可直接复现的层级方向 recipe。

## 11. Wang et al. 2025 - Refusal Direction is Universal Across Safety-Aligned Languages

PDF：`2025_Wang_Multilingual_Refusal_Direction.pdf`

链接：https://arxiv.org/abs/2505.17306

核心结论：拒绝方向不只是英语安全行为中的局部现象，而是在多种安全对齐语言之间具有很强迁移性。英文抽取的 refusal direction 可以迁移到其他语言，其他安全对齐语言中抽取的方向也能相互迁移。

方法：

- 构建或使用多语言 harmful/benign prompt 对照集。
- 在不同语言上抽取 refusal direction。
- 做跨语言方向迁移和 activation intervention。
- 分析不同语言 refusal vectors 的平行性。

对 CompSP 的意义：

- 这篇论文给“跨主题泛化”提供了一个很好的白盒类比：如果 refusal direction 能跨语言共享，那么 CompSP 的跨主题泛化也可能来自某种共享安全坐标。
- 但它研究的是同一白盒模型内的跨语言，不等价于黑盒目标模型之间共享结构。写作时只能作为机制类比，不能直接当作跨模型证据。

可复用指标：

- 不同主题/攻击方案下方向 cosine。
- 按攻击主题抽取方向，再测试跨主题迁移。
- 如果后续加入中文/英文 prompt，可测试 CompSP 排序是否也有跨语言稳定性。

## 12. Piras et al. 2025/2026 - Multi-Directional Refusal Suppression in Language Models

PDF：`2025_Piras_Multi_Directional_Refusal_Suppression.pdf`

链接：https://arxiv.org/abs/2511.08379

核心结论：拒绝行为不一定只由单个方向充分描述，更像是由多个相关方向共同组成的低维 manifold。多方向方法能比单方向更细粒度地压制或分析拒绝。

方法：

- 在 harmful prompt 表示上训练 Self-Organizing Map。
- 用 SOM 节点中心构造多个 refusal directions。
- 比较单方向与多方向抑制 refusal 的效果。

对 CompSP 的意义：

- 如果单方向不能解释 CompSP 错排，多方向 refusal manifold 是自然升级路线。
- OFA、PAIR、DrAttack 的 prompt 形式差异很大，可能落在 refusal manifold 的不同局部区域；单方向平均会抹掉这些局部结构。

可复用指标：

- 每个 prompt 到多个 refusal directions 的最大投影、均值投影、top-k 投影向量。
- prompt 在多方向空间中的聚类，与 CompSP 排名桶或攻击方案标签比较。

## 13. Joad et al. 2026 - There Is More to Refusal in Large Language Models than a Single Direction

PDF：`2026_Joad_More_Than_Single_Refusal_Direction.pdf`

链接：https://arxiv.org/abs/2602.02132

核心结论：拒绝不是单一同质行为。安全拒绝、能力不足、过度拒绝、拟人化拒绝等不同拒绝类型在激活空间中对应几何上不同的方向；不过不同方向的线性 steering 可能共享一个类似的一维 refusal-control trade-off。

方法：

- 构造多种 refusal / non-compliance 类别。
- 分别抽取类别方向。
- 比较方向几何关系和 steering 后的拒绝-过度拒绝权衡。

对 CompSP 的意义：

- 这篇文献提醒我们：ALR 长回答/拒绝行为并不只有“安全拒绝”一种来源。
- CompSP 可能学到某种更宽泛的 non-compliance 或 safety style 结构，而不是严格的 harmfulness 或 refusal。

可复用指标：

- 分别构造 safety refusal、over-refusal、capability refusal 的方向。
- 对错排案例看它们更接近哪类 refusal 方向。
- 检查低 ASR 高 CompSP 的 prompt 是否触发了“非安全型拒绝”方向。

## 14. Cristofano 2026 - Universal Refusal Circuits Across LLMs

PDF：`2026_Cristofano_Universal_Refusal_Circuits.pdf`

链接：https://arxiv.org/abs/2601.16034

核心结论：拒绝行为可能存在跨模型共享的低维语义 circuit。论文尝试把 donor model 的 refusal intervention 通过 concept-basis reconstruction 转移到不同架构或训练方式的 target model。

方法：

- 用 concept fingerprints 对齐不同模型的层。
- 用概念基重构 donor refusal trajectory。
- 在 target model 上重放或迁移 refusal ablation。
- 加入 SVD 稳定性约束减少能力损伤。

对 CompSP 的意义：

- 这篇文章和我们的“黑盒代理是否学到白盒结构”非常接近，但它仍然是白盒/半白盒转移。
- 可以作为强叙事背景：安全结构可能跨模型共享；但我们的实验必须避免直接把 Llama 白盒方向等同于 Qwen 或商业黑盒方向。

可复用指标：

- 跨模型方向/子空间对齐的概念框架。
- 如果后续有 Qwen 开源白盒模型，可做 Llama/Qwen 白盒结构对照，再和 CompSP 跨模型结果比较。

## 更新后的优先阅读顺序

1. Arditi 单方向拒绝论文。
2. Zhao 有害性/拒绝分离论文。
3. Pan 多维安全方向论文。
4. Piras/Joad 多方向 refusal 论文。
5. Wang 多语言拒绝方向论文。
6. Lin 越狱表示空间分析论文。
7. SCANS，作为可直接复现的层级方向 recipe。
8. Cristofano 跨模型 refusal circuits，作为远期叙事参考。
