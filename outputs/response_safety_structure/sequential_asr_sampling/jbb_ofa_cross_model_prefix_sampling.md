# 序贯 ASR 增采离线模拟

选择比例：top 20%；高风险阈值：future ASR >= 0.5。

## 前缀回答数 k=1

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.638 | -0.000 | 0.649 | 0.200 | 0.206 |
| prefix_empirical_asr | 0.751 | 0.113 | 0.776 | 0.239 | 0.004 |
| prefix_jeffreys_asr | 0.751 | 0.113 | 0.776 | 0.239 | 0.003 |
| prefix_alr | 0.705 | 0.067 | 0.721 | 0.222 | 0.188 |
| prefix_judge_hazard | 0.759 | 0.121 | 0.785 | 0.242 | 0.004 |
| prefix_hazard_weighted_asr | 0.759 | 0.121 | 0.785 | 0.242 | 0.004 |

## 前缀回答数 k=2

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.638 | -0.000 | 0.657 | 0.200 | 0.210 |
| prefix_empirical_asr | 0.775 | 0.137 | 0.812 | 0.247 | 0.004 |
| prefix_jeffreys_asr | 0.775 | 0.137 | 0.812 | 0.247 | 0.002 |
| prefix_alr | 0.715 | 0.077 | 0.742 | 0.226 | 0.184 |
| prefix_judge_hazard | 0.766 | 0.128 | 0.797 | 0.243 | 0.004 |
| prefix_hazard_weighted_asr | 0.782 | 0.144 | 0.815 | 0.248 | 0.002 |

## 前缀回答数 k=4

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.638 | -0.000 | 0.657 | 0.200 | 0.198 |
| prefix_empirical_asr | 0.794 | 0.156 | 0.830 | 0.252 | 0.007 |
| prefix_jeffreys_asr | 0.794 | 0.156 | 0.830 | 0.252 | 0.000 |
| prefix_alr | 0.722 | 0.084 | 0.749 | 0.228 | 0.192 |
| prefix_judge_hazard | 0.770 | 0.132 | 0.797 | 0.243 | 0.004 |
| prefix_hazard_weighted_asr | 0.800 | 0.162 | 0.830 | 0.253 | 0.002 |

