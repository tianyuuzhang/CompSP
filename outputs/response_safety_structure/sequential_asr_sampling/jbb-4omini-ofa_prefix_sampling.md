# 序贯 ASR 增采离线模拟

选择比例：top 20%；高风险阈值：future ASR >= 0.5。

## 前缀回答数 k=1

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.692 | 0.002 | 0.721 | 0.200 | 0.205 |
| prefix_empirical_asr | 0.772 | 0.082 | 0.810 | 0.225 | 0.001 |
| prefix_jeffreys_asr | 0.772 | 0.081 | 0.812 | 0.226 | 0.002 |
| prefix_alr | 0.761 | 0.070 | 0.794 | 0.221 | 0.186 |
| prefix_judge_hazard | 0.777 | 0.087 | 0.817 | 0.227 | 0.001 |
| prefix_hazard_weighted_asr | 0.777 | 0.087 | 0.815 | 0.227 | 0.001 |

## 前缀回答数 k=2

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.692 | 0.002 | 0.730 | 0.201 | 0.215 |
| prefix_empirical_asr | 0.787 | 0.097 | 0.835 | 0.230 | 0.001 |
| prefix_jeffreys_asr | 0.786 | 0.096 | 0.836 | 0.230 | 0.002 |
| prefix_alr | 0.767 | 0.076 | 0.808 | 0.222 | 0.177 |
| prefix_judge_hazard | 0.783 | 0.093 | 0.830 | 0.228 | 0.000 |
| prefix_hazard_weighted_asr | 0.793 | 0.103 | 0.839 | 0.231 | 0.004 |

## 前缀回答数 k=4

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.692 | 0.002 | 0.731 | 0.201 | 0.262 |
| prefix_empirical_asr | 0.801 | 0.111 | 0.850 | 0.233 | 0.006 |
| prefix_jeffreys_asr | 0.800 | 0.110 | 0.850 | 0.233 | 0.006 |
| prefix_alr | 0.771 | 0.081 | 0.812 | 0.223 | 0.203 |
| prefix_judge_hazard | 0.785 | 0.095 | 0.829 | 0.227 | 0.000 |
| prefix_hazard_weighted_asr | 0.809 | 0.119 | 0.851 | 0.234 | 0.005 |

