# 前缀回答文本残差驱动的序贯 ASR 增采

选择比例：top 20%；高风险阈值：future ASR >= 0.5。
文本清理：`mask_refusal_hazard_terms`。

## 前缀回答数 k=1

训练记录均值：17861；测试记录均值：6000。

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.627 | -0.000 | 0.644 | 0.201 | 0.179 |
| prefix_hazard_weighted_asr | 0.744 | 0.116 | 0.767 | 0.239 | 0.000 |
| prefix_alr | 0.728 | 0.100 | 0.748 | 0.233 | 0.157 |
| model_base_features | 0.739 | 0.111 | 0.762 | 0.237 | 0.197 |
| model_text_only | 0.755 | 0.127 | 0.774 | 0.241 | 0.144 |
| model_text_residual | 0.742 | 0.114 | 0.759 | 0.237 | 0.132 |
| model_base_plus_text_residual | 0.751 | 0.124 | 0.773 | 0.241 | 0.182 |

## 前缀回答数 k=2

训练记录均值：17861；测试记录均值：6000。

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.628 | 0.000 | 0.652 | 0.202 | 0.257 |
| prefix_hazard_weighted_asr | 0.773 | 0.146 | 0.797 | 0.246 | 0.000 |
| prefix_alr | 0.746 | 0.119 | 0.776 | 0.240 | 0.178 |
| model_base_features | 0.746 | 0.119 | 0.777 | 0.240 | 0.178 |
| model_text_only | 0.767 | 0.140 | 0.791 | 0.245 | 0.089 |
| model_text_residual | 0.748 | 0.120 | 0.767 | 0.237 | 0.168 |
| model_base_plus_text_residual | 0.760 | 0.132 | 0.785 | 0.243 | 0.168 |

## 前缀回答数 k=4

训练记录均值：17861；测试记录均值：6000。

| 策略 | selected future ASR | lift | 高风险precision | 高风险recall | prefix零但后缀高风险recall |
|---|---:|---:|---:|---:|---:|
| random | 0.628 | 0.001 | 0.652 | 0.201 | 0.375 |
| prefix_hazard_weighted_asr | 0.791 | 0.164 | 0.813 | 0.250 | 0.000 |
| prefix_alr | 0.754 | 0.126 | 0.781 | 0.240 | 0.188 |
| model_base_features | 0.751 | 0.124 | 0.785 | 0.242 | 0.188 |
| model_text_only | 0.767 | 0.140 | 0.791 | 0.244 | 0.062 |
| model_text_residual | 0.757 | 0.130 | 0.777 | 0.239 | 0.062 |
| model_base_plus_text_residual | 0.763 | 0.136 | 0.788 | 0.243 | 0.188 |

