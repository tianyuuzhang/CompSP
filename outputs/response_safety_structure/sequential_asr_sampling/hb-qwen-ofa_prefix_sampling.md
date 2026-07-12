# 序贯 ASR 增采离线模拟

选择比例：top 20%；高风险阈值：future ASR >= 0.5。

## 前缀回答数 k=1

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.570 | 0.006 | 0.601 | 0.202 | 0.196 |
| prefix_empirical_asr | 0.701 | 0.137 | 0.749 | 0.251 | 0.001 |
| prefix_jeffreys_asr | 0.699 | 0.135 | 0.746 | 0.250 | 0.000 |
| prefix_alr | 0.642 | 0.078 | 0.681 | 0.228 | 0.198 |
| prefix_judge_hazard | 0.710 | 0.146 | 0.758 | 0.254 | 0.001 |
| prefix_hazard_weighted_asr | 0.712 | 0.147 | 0.760 | 0.255 | 0.001 |

## 前缀回答数 k=2

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.570 | 0.006 | 0.612 | 0.202 | 0.192 |
| prefix_empirical_asr | 0.720 | 0.156 | 0.778 | 0.257 | 0.000 |
| prefix_jeffreys_asr | 0.718 | 0.154 | 0.777 | 0.257 | 0.000 |
| prefix_alr | 0.656 | 0.091 | 0.704 | 0.232 | 0.226 |
| prefix_judge_hazard | 0.723 | 0.159 | 0.778 | 0.257 | 0.000 |
| prefix_hazard_weighted_asr | 0.729 | 0.165 | 0.784 | 0.259 | 0.000 |

## 前缀回答数 k=4

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.570 | 0.006 | 0.616 | 0.203 | 0.261 |
| prefix_empirical_asr | 0.740 | 0.176 | 0.793 | 0.261 | 0.000 |
| prefix_jeffreys_asr | 0.738 | 0.173 | 0.793 | 0.261 | 0.000 |
| prefix_alr | 0.663 | 0.098 | 0.715 | 0.236 | 0.223 |
| prefix_judge_hazard | 0.736 | 0.172 | 0.781 | 0.258 | 0.000 |
| prefix_hazard_weighted_asr | 0.750 | 0.185 | 0.797 | 0.263 | 0.000 |

