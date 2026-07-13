# 固定预算追加采样模拟

初始每条观察 k 次；额外预算给同题 top 20% 候选；清理口径：`mask_strong_artifacts`。

## k=1

| 策略 | extra/row | final recall | recall gain | prefix-zero高风险recall | selected future ASR |
|---|---:|---:|---:|---:|---:|
| random | 3.80 | 0.827 | 0.042 | 0.193 | 0.630 |
| prefix_hazard_weighted_asr | 3.80 | 0.793 | 0.008 | 0.003 | 0.752 |
| prefix_alr | 3.80 | 0.820 | 0.035 | 0.167 | 0.727 |
| model_base_features | 3.80 | 0.822 | 0.037 | 0.199 | 0.734 |
| model_text_only | 3.80 | 0.818 | 0.032 | 0.156 | 0.754 |
| model_base_plus_text_residual | 3.80 | 0.820 | 0.035 | 0.173 | 0.745 |
| hybrid_hazard_text_q0.25 | 3.80 | 0.830 | 0.045 | 0.390 | 0.686 |
| hybrid_hazard_text_q0.5 | 3.80 | 0.860 | 0.074 | 0.611 | 0.616 |

## k=2

| 策略 | extra/row | final recall | recall gain | prefix-zero高风险recall | selected future ASR |
|---|---:|---:|---:|---:|---:|
| random | 3.60 | 0.891 | 0.027 | 0.204 | 0.630 |
| prefix_hazard_weighted_asr | 3.60 | 0.865 | 0.001 | 0.000 | 0.774 |
| prefix_alr | 3.60 | 0.886 | 0.022 | 0.171 | 0.742 |
| model_base_features | 3.60 | 0.887 | 0.023 | 0.191 | 0.746 |
| model_text_only | 3.60 | 0.884 | 0.020 | 0.116 | 0.763 |
| model_base_plus_text_residual | 3.60 | 0.886 | 0.022 | 0.152 | 0.756 |
| hybrid_hazard_text_q0.25 | 3.60 | 0.896 | 0.032 | 0.552 | 0.683 |
| hybrid_hazard_text_q0.5 | 3.60 | 0.914 | 0.050 | 0.749 | 0.604 |

## k=4

| 策略 | extra/row | final recall | recall gain | prefix-zero高风险recall | selected future ASR |
|---|---:|---:|---:|---:|---:|
| random | 3.20 | 0.935 | 0.016 | 0.215 | 0.630 |
| prefix_hazard_weighted_asr | 3.20 | 0.919 | 0.000 | 0.000 | 0.792 |
| prefix_alr | 3.20 | 0.933 | 0.014 | 0.227 | 0.750 |
| model_base_features | 3.20 | 0.933 | 0.014 | 0.222 | 0.750 |
| model_text_only | 3.20 | 0.931 | 0.012 | 0.182 | 0.768 |
| model_base_plus_text_residual | 3.20 | 0.932 | 0.013 | 0.192 | 0.762 |
| hybrid_hazard_text_q0.25 | 3.20 | 0.937 | 0.018 | 0.633 | 0.690 |
| hybrid_hazard_text_q0.5 | 3.20 | 0.947 | 0.028 | 0.685 | 0.622 |

