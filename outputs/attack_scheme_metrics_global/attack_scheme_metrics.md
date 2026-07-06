# 攻击方案 IASR/FASC 提升实验

随机 baseline：每个问题在全局 100 条候选上随机排序 `10000` 次，IASR 取全局前 20 条，FASC 按全局随机序列计算首次成功成本。

Ranked 计算：Pair 和 DrAttack 仍使用已有 BTL 四段排名，每段取 Top-5；这是因为 CompSP pairwise 比较是 `n^2` 复杂度，实验中按 `k=25` 分组。

FASC 计算：按排序顺序访问，`tx=0` 记 20 次成本，遇到第一个 `tx>0` 记 `20/tx` 后停止。

## 结果

| 攻击 | 数据集 | 模型 | Random 问题数 | 排名组数 | Random IASR | Ranked IASR | IASR 提升 | Random FASC | Ranked FASC | FASC 降低 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ofa | jbb | 4omini | 20 | 0 | 70.37 | NA | NA | 12.64 | NA | NA |
| ofa | hb | 4omini | 19 | 0 | 42.46 | NA | NA | 67.45 | NA | NA |
| pair | jbb | 4omini | 20 | 80 | 62.40 | 75.10 | 12.70 | 107.77 | 27.42 | 80.35 |
| drattack | jbb | 4omini | 20 | 80 | 56.99 | 64.12 | 7.14 | 23.54 | 13.42 | 10.13 |
| ofa | jbb | llama | 20 | 0 | 65.15 | NA | NA | 10.92 | NA | NA |
| ofa | hb | llama | 19 | 0 | 35.07 | NA | NA | 27.15 | NA | NA |
| pair | jbb | llama | 20 | 80 | 69.70 | 80.47 | 10.77 | 106.37 | 27.02 | 79.35 |
| drattack | jbb | llama | 20 | 80 | 53.44 | 58.42 | 4.98 | 50.64 | 27.97 | 22.67 |
| ofa | jbb | qwen | 20 | 0 | 40.33 | NA | NA | 40.55 | NA | NA |
| ofa | hb | qwen | 20 | 0 | 29.95 | NA | NA | 125.15 | NA | NA |
| pair | jbb | qwen | 20 | 80 | 56.44 | 74.89 | 18.45 | 109.08 | 27.96 | 81.12 |
| drattack | jbb | qwen | 20 | 80 | 48.79 | 55.64 | 6.85 | 13.90 | 7.27 | 6.63 |

## OFA 与论文表格对照

当前目录未发现 OFA 对应的 `data/saverk` BTL 排名，因此 OFA 只复算 random baseline，并与前期论文表格中的 CompSP 结果对照。

| 数据集 | 模型 | 本次 Random IASR | 论文 Random IASR | 差值 | 本次 Random FASC | 论文 Random FASC | 差值 | 论文 CompSP IASR | 论文 CompSP FASC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| jbb | 4omini | 70.37 | 70.40 | -0.03 | 12.64 | 12.70 | -0.06 | 79.70 | 1.80 |
| hb | 4omini | 42.46 | 42.50 | -0.04 | 67.45 | 63.30 | 4.15 | 49.70 | 14.70 |
| jbb | llama | 65.15 | 65.20 | -0.05 | 10.92 | 10.90 | 0.02 | 79.20 | 1.40 |
| hb | llama | 35.07 | 35.60 | -0.53 | 27.15 | 25.40 | 1.75 | 48.50 | 10.10 |
| jbb | qwen | 40.33 | 40.30 | 0.03 | 40.55 | 40.70 | -0.15 | 57.80 | 11.90 |
| hb | qwen | 29.95 | 29.90 | 0.05 | 125.15 | 127.70 | -2.55 | 41.60 | 16.20 |

## 注意事项

- Pair 和 DrAttack 的 Ranked 指标来自 `data/saverk` 中已有 BTL 排名。
- OFA 的 CompSP 排名文件当前未定位到，本报告不伪造重算结果，只引用论文表格作为对照。
- 如果后续补齐 OFA 的 BTL 排名，可以用同一脚本直接重算 OFA 的 Ranked IASR/FASC。
