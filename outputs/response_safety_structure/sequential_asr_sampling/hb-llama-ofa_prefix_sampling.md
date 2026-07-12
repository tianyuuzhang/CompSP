# 序贯 ASR 增采离线模拟

选择比例：top 20%；高风险阈值：future ASR >= 0.5。

## 前缀回答数 k=1

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.663 | 0.003 | 0.703 | 0.200 | 0.200 |
| prefix_empirical_asr | 0.783 | 0.123 | 0.836 | 0.238 | 0.000 |
| prefix_jeffreys_asr | 0.776 | 0.116 | 0.828 | 0.236 | 0.001 |
| prefix_alr | 0.768 | 0.108 | 0.820 | 0.234 | 0.167 |
| prefix_judge_hazard | 0.782 | 0.122 | 0.826 | 0.235 | 0.000 |
| prefix_hazard_weighted_asr | 0.782 | 0.122 | 0.828 | 0.236 | 0.002 |

## 前缀回答数 k=2

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.662 | 0.002 | 0.711 | 0.200 | 0.176 |
| prefix_empirical_asr | 0.804 | 0.144 | 0.858 | 0.242 | 0.000 |
| prefix_jeffreys_asr | 0.799 | 0.139 | 0.855 | 0.241 | 0.000 |
| prefix_alr | 0.786 | 0.127 | 0.840 | 0.237 | 0.145 |
| prefix_judge_hazard | 0.780 | 0.120 | 0.820 | 0.231 | 0.000 |
| prefix_hazard_weighted_asr | 0.813 | 0.153 | 0.857 | 0.241 | 0.000 |

## 前缀回答数 k=4

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.662 | 0.002 | 0.705 | 0.199 | 0.174 |
| prefix_empirical_asr | 0.829 | 0.169 | 0.870 | 0.245 | 0.000 |
| prefix_jeffreys_asr | 0.825 | 0.165 | 0.871 | 0.246 | 0.000 |
| prefix_alr | 0.802 | 0.142 | 0.856 | 0.241 | 0.079 |
| prefix_judge_hazard | 0.767 | 0.107 | 0.797 | 0.225 | 0.000 |
| prefix_hazard_weighted_asr | 0.834 | 0.174 | 0.868 | 0.245 | 0.000 |

