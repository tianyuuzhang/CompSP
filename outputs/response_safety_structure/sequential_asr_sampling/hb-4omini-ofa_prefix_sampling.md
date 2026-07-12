# 序贯 ASR 增采离线模拟

选择比例：top 20%；高风险阈值：future ASR >= 0.5。

## 前缀回答数 k=1

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.673 | 0.003 | 0.710 | 0.200 | 0.202 |
| prefix_empirical_asr | 0.761 | 0.090 | 0.808 | 0.228 | 0.004 |
| prefix_jeffreys_asr | 0.758 | 0.088 | 0.803 | 0.226 | 0.002 |
| prefix_alr | 0.750 | 0.080 | 0.798 | 0.225 | 0.181 |
| prefix_judge_hazard | 0.764 | 0.094 | 0.808 | 0.228 | 0.003 |
| prefix_hazard_weighted_asr | 0.760 | 0.089 | 0.808 | 0.228 | 0.001 |

## 前缀回答数 k=2

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.673 | 0.002 | 0.723 | 0.201 | 0.203 |
| prefix_empirical_asr | 0.776 | 0.105 | 0.833 | 0.231 | 0.003 |
| prefix_jeffreys_asr | 0.778 | 0.107 | 0.834 | 0.231 | 0.000 |
| prefix_alr | 0.762 | 0.091 | 0.820 | 0.228 | 0.177 |
| prefix_judge_hazard | 0.767 | 0.096 | 0.820 | 0.227 | 0.003 |
| prefix_hazard_weighted_asr | 0.785 | 0.114 | 0.839 | 0.233 | 0.000 |

## 前缀回答数 k=4

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.672 | 0.003 | 0.725 | 0.201 | 0.258 |
| prefix_empirical_asr | 0.789 | 0.119 | 0.844 | 0.234 | 0.000 |
| prefix_jeffreys_asr | 0.790 | 0.120 | 0.843 | 0.234 | 0.000 |
| prefix_alr | 0.765 | 0.095 | 0.821 | 0.228 | 0.204 |
| prefix_judge_hazard | 0.765 | 0.096 | 0.807 | 0.224 | 0.000 |
| prefix_hazard_weighted_asr | 0.798 | 0.129 | 0.844 | 0.234 | 0.000 |

