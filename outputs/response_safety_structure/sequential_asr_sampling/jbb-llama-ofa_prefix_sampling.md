# 序贯 ASR 增采离线模拟

选择比例：top 20%；高风险阈值：future ASR >= 0.5。

## 前缀回答数 k=1

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.636 | 0.001 | 0.643 | 0.200 | 0.203 |
| prefix_empirical_asr | 0.763 | 0.128 | 0.791 | 0.246 | 0.004 |
| prefix_jeffreys_asr | 0.763 | 0.128 | 0.789 | 0.245 | 0.003 |
| prefix_alr | 0.747 | 0.112 | 0.769 | 0.239 | 0.166 |
| prefix_judge_hazard | 0.772 | 0.137 | 0.797 | 0.248 | 0.003 |
| prefix_hazard_weighted_asr | 0.771 | 0.136 | 0.796 | 0.248 | 0.002 |

## 前缀回答数 k=2

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.636 | 0.001 | 0.653 | 0.200 | 0.192 |
| prefix_empirical_asr | 0.793 | 0.158 | 0.837 | 0.257 | 0.001 |
| prefix_jeffreys_asr | 0.794 | 0.159 | 0.836 | 0.256 | 0.000 |
| prefix_alr | 0.770 | 0.135 | 0.806 | 0.247 | 0.156 |
| prefix_judge_hazard | 0.765 | 0.130 | 0.794 | 0.244 | 0.000 |
| prefix_hazard_weighted_asr | 0.803 | 0.168 | 0.840 | 0.258 | 0.001 |

## 前缀回答数 k=4

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.636 | 0.001 | 0.655 | 0.201 | 0.182 |
| prefix_empirical_asr | 0.818 | 0.183 | 0.855 | 0.262 | 0.000 |
| prefix_jeffreys_asr | 0.819 | 0.183 | 0.858 | 0.263 | 0.000 |
| prefix_alr | 0.787 | 0.152 | 0.824 | 0.252 | 0.118 |
| prefix_judge_hazard | 0.757 | 0.122 | 0.778 | 0.238 | 0.000 |
| prefix_hazard_weighted_asr | 0.823 | 0.188 | 0.857 | 0.262 | 0.000 |

