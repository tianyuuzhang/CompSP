# 序贯 ASR 增采离线模拟

选择比例：top 20%；高风险阈值：future ASR >= 0.5。

## 前缀回答数 k=1

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.688 | -0.002 | 0.724 | 0.199 | 0.195 |
| prefix_empirical_asr | 0.783 | 0.093 | 0.832 | 0.229 | 0.001 |
| prefix_jeffreys_asr | 0.783 | 0.093 | 0.830 | 0.228 | 0.001 |
| prefix_alr | 0.757 | 0.068 | 0.800 | 0.220 | 0.200 |
| prefix_judge_hazard | 0.793 | 0.103 | 0.839 | 0.231 | 0.001 |
| prefix_hazard_weighted_asr | 0.791 | 0.101 | 0.836 | 0.230 | 0.000 |

## 前缀回答数 k=2

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.688 | -0.002 | 0.732 | 0.199 | 0.186 |
| prefix_empirical_asr | 0.803 | 0.114 | 0.859 | 0.233 | 0.001 |
| prefix_jeffreys_asr | 0.805 | 0.115 | 0.862 | 0.234 | 0.000 |
| prefix_alr | 0.767 | 0.078 | 0.820 | 0.223 | 0.192 |
| prefix_judge_hazard | 0.802 | 0.112 | 0.851 | 0.231 | 0.000 |
| prefix_hazard_weighted_asr | 0.814 | 0.125 | 0.865 | 0.235 | 0.000 |

## 前缀回答数 k=4

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.687 | -0.002 | 0.731 | 0.199 | 0.182 |
| prefix_empirical_asr | 0.822 | 0.133 | 0.873 | 0.237 | 0.000 |
| prefix_jeffreys_asr | 0.823 | 0.133 | 0.874 | 0.237 | 0.000 |
| prefix_alr | 0.775 | 0.086 | 0.827 | 0.225 | 0.223 |
| prefix_judge_hazard | 0.804 | 0.115 | 0.846 | 0.230 | 0.000 |
| prefix_hazard_weighted_asr | 0.832 | 0.143 | 0.875 | 0.238 | 0.000 |

