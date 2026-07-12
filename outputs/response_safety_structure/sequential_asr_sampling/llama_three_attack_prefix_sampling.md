# 序贯 ASR 增采离线模拟

选择比例：top 20%；高风险阈值：future ASR >= 0.5。

## 前缀回答数 k=1

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.636 | 0.000 | 0.644 | 0.200 | 0.200 |
| prefix_empirical_asr | 0.758 | 0.123 | 0.786 | 0.244 | 0.002 |
| prefix_jeffreys_asr | 0.761 | 0.126 | 0.789 | 0.245 | 0.003 |
| prefix_alr | 0.738 | 0.103 | 0.758 | 0.235 | 0.168 |
| prefix_judge_hazard | 0.771 | 0.136 | 0.800 | 0.248 | 0.002 |
| prefix_hazard_weighted_asr | 0.770 | 0.135 | 0.799 | 0.248 | 0.003 |

## 前缀回答数 k=2

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.636 | 0.000 | 0.654 | 0.200 | 0.202 |
| prefix_empirical_asr | 0.791 | 0.156 | 0.834 | 0.255 | 0.001 |
| prefix_jeffreys_asr | 0.791 | 0.155 | 0.832 | 0.254 | 0.002 |
| prefix_alr | 0.757 | 0.121 | 0.789 | 0.241 | 0.155 |
| prefix_judge_hazard | 0.773 | 0.138 | 0.806 | 0.247 | 0.000 |
| prefix_hazard_weighted_asr | 0.802 | 0.167 | 0.839 | 0.257 | 0.002 |

## 前缀回答数 k=4

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.636 | 0.000 | 0.655 | 0.200 | 0.207 |
| prefix_empirical_asr | 0.816 | 0.180 | 0.855 | 0.261 | 0.003 |
| prefix_jeffreys_asr | 0.816 | 0.181 | 0.856 | 0.261 | 0.002 |
| prefix_alr | 0.771 | 0.136 | 0.804 | 0.246 | 0.124 |
| prefix_judge_hazard | 0.772 | 0.137 | 0.800 | 0.244 | 0.002 |
| prefix_hazard_weighted_asr | 0.824 | 0.188 | 0.858 | 0.262 | 0.004 |

