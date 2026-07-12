# 序贯 ASR 增采离线模拟

选择比例：top 20%；高风险阈值：future ASR >= 0.5。

## 前缀回答数 k=1

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.842 | 0.002 | 0.857 | 0.200 | 0.198 |
| prefix_empirical_asr | 0.896 | 0.056 | 0.919 | 0.215 | 0.001 |
| prefix_jeffreys_asr | 0.895 | 0.054 | 0.917 | 0.214 | 0.002 |
| prefix_alr | 0.848 | 0.008 | 0.864 | 0.202 | 0.199 |
| prefix_judge_hazard | 0.902 | 0.061 | 0.925 | 0.216 | 0.002 |
| prefix_hazard_weighted_asr | 0.901 | 0.061 | 0.926 | 0.217 | 0.001 |

## 前缀回答数 k=2

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.842 | 0.002 | 0.862 | 0.201 | 0.241 |
| prefix_empirical_asr | 0.911 | 0.070 | 0.937 | 0.218 | 0.000 |
| prefix_jeffreys_asr | 0.910 | 0.069 | 0.937 | 0.218 | 0.000 |
| prefix_alr | 0.848 | 0.008 | 0.869 | 0.202 | 0.194 |
| prefix_judge_hazard | 0.912 | 0.072 | 0.938 | 0.218 | 0.000 |
| prefix_hazard_weighted_asr | 0.913 | 0.072 | 0.938 | 0.218 | 0.000 |

## 前缀回答数 k=4

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.842 | 0.002 | 0.863 | 0.201 | 0.162 |
| prefix_empirical_asr | 0.922 | 0.081 | 0.946 | 0.220 | 0.000 |
| prefix_jeffreys_asr | 0.921 | 0.081 | 0.946 | 0.220 | 0.000 |
| prefix_alr | 0.849 | 0.008 | 0.872 | 0.203 | 0.180 |
| prefix_judge_hazard | 0.920 | 0.080 | 0.943 | 0.219 | 0.000 |
| prefix_hazard_weighted_asr | 0.922 | 0.082 | 0.945 | 0.220 | 0.000 |

