# 序贯 ASR 增采离线模拟

选择比例：top 20%；高风险阈值：future ASR >= 0.5。

## 前缀回答数 k=1

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.386 | -0.001 | 0.378 | 0.200 | 0.215 |
| prefix_empirical_asr | 0.575 | 0.188 | 0.589 | 0.312 | 0.015 |
| prefix_jeffreys_asr | 0.572 | 0.185 | 0.586 | 0.310 | 0.013 |
| prefix_alr | 0.466 | 0.080 | 0.464 | 0.245 | 0.214 |
| prefix_judge_hazard | 0.586 | 0.200 | 0.601 | 0.318 | 0.014 |
| prefix_hazard_weighted_asr | 0.585 | 0.199 | 0.600 | 0.317 | 0.012 |

## 前缀回答数 k=2

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.386 | -0.001 | 0.386 | 0.200 | 0.215 |
| prefix_empirical_asr | 0.611 | 0.225 | 0.641 | 0.332 | 0.004 |
| prefix_jeffreys_asr | 0.611 | 0.225 | 0.640 | 0.332 | 0.012 |
| prefix_alr | 0.475 | 0.089 | 0.483 | 0.250 | 0.184 |
| prefix_judge_hazard | 0.605 | 0.218 | 0.628 | 0.326 | 0.009 |
| prefix_hazard_weighted_asr | 0.620 | 0.233 | 0.646 | 0.335 | 0.004 |

## 前缀回答数 k=4

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.386 | -0.001 | 0.387 | 0.200 | 0.220 |
| prefix_empirical_asr | 0.637 | 0.251 | 0.667 | 0.344 | 0.000 |
| prefix_jeffreys_asr | 0.636 | 0.250 | 0.665 | 0.344 | 0.018 |
| prefix_alr | 0.480 | 0.094 | 0.488 | 0.252 | 0.180 |
| prefix_judge_hazard | 0.616 | 0.230 | 0.638 | 0.330 | 0.005 |
| prefix_hazard_weighted_asr | 0.644 | 0.258 | 0.669 | 0.346 | 0.000 |

