# 数据索引

## 原始问题文件

- `/remote-home/zty/data/jailbreak_questions.json`：JBB 抽样出的原始有害问题列表，大小约 9,401 字节。
- `/remote-home/zty/data/harmbench_questions.json`：HB 抽样出的原始有害问题列表，大小约 37,198 字节。
- `/remote-home/zty/data/data_jailbreakbench.csv`：早期 JBB CSV 来源，用于本地 vLLM 脚本。
- `/remote-home/zty/data/harmbench_behaviors_text_all.csv`：HB 行为来源。

## 已整理的数据

- `/remote-home/zty/tidy_CompSP/data/原始问题/`：复制过来的 JBB/HB 原始问题 JSON。
- `/remote-home/zty/tidy_CompSP/data/大纲拆分/`：q0/a0/q1 的大纲拆分输出，按源模型和数据集分组。
- `/remote-home/zty/tidy_CompSP/data/saverk/`：CompSP 用的 pairwise matrix 和 BTL ranking 输出。

## 分析产物

- `/remote-home/zty/tidy_CompSP/data_ana/总体含毒比例分析`
- `/remote-home/zty/tidy_CompSP/data_ana/模型回答长度分布分析`
- `/remote-home/zty/tidy_CompSP/data_ana/不同模型对同指令ASR相关性`
- `/remote-home/zty/tidy_CompSP/data_ana/ASR-ALR跨模型相关性`
- `/remote-home/zty/tidy_CompSP/data_ana/回归预测baseline`
- `/remote-home/zty/tidy_CompSP/data_ana/预测成功率随ASR`
- `/remote-home/zty/tidy_CompSP/data_ana/预测成功率随ASR/ALR分布`

## 外部数据依赖

- `/remote-home/xzh/for_zty/reproduce/tasks/pair/...`
- `/remote-home/xzh/for_zty/reproduce/drattack/DrAttack/Instructions_rewrite...`
- `/remote-home/zty/save/...`：原始生成实验输出，没有搬进清理后的树。
