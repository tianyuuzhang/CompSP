# 序贯 ASR 增采离线模拟

选择比例：top 20%；高风险阈值：future ASR >= 0.5。

## 前缀回答数 k=1

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.860 | -0.004 | 0.893 | 0.198 | 0.188 |
| prefix_empirical_asr | 0.905 | 0.042 | 0.949 | 0.211 | 0.000 |
| prefix_jeffreys_asr | 0.902 | 0.038 | 0.948 | 0.211 | 0.000 |
| prefix_alr | 0.864 | 0.000 | 0.903 | 0.201 | 0.200 |
| prefix_judge_hazard | 0.913 | 0.049 | 0.951 | 0.211 | 0.000 |
| prefix_hazard_weighted_asr | 0.909 | 0.045 | 0.945 | 0.210 | 0.000 |

## 前缀回答数 k=2

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.859 | -0.004 | 0.899 | 0.198 | 0.171 |
| prefix_empirical_asr | 0.925 | 0.061 | 0.975 | 0.215 | 0.000 |
| prefix_jeffreys_asr | 0.922 | 0.058 | 0.976 | 0.215 | 0.000 |
| prefix_alr | 0.865 | 0.002 | 0.910 | 0.201 | 0.182 |
| prefix_judge_hazard | 0.933 | 0.070 | 0.981 | 0.216 | 0.000 |
| prefix_hazard_weighted_asr | 0.933 | 0.070 | 0.981 | 0.216 | 0.000 |

## 前缀回答数 k=4

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.860 | -0.004 | 0.902 | 0.199 | 0.103 |
| prefix_empirical_asr | 0.940 | 0.076 | 0.988 | 0.218 | 0.000 |
| prefix_jeffreys_asr | 0.937 | 0.073 | 0.986 | 0.217 | 0.000 |
| prefix_alr | 0.865 | 0.002 | 0.913 | 0.201 | 0.166 |
| prefix_judge_hazard | 0.947 | 0.083 | 0.990 | 0.218 | 0.000 |
| prefix_hazard_weighted_asr | 0.947 | 0.084 | 0.991 | 0.218 | 0.000 |

