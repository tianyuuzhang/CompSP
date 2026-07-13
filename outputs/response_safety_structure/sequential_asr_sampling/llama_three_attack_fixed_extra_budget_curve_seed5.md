# 固定 extra response 预算的 ASR 发现曲线

每条先观察前缀 `k` 次；每个数据集-问题组按 `extra_budget_per_row` 获得相同追加访问预算；清理口径：`mask_strong_artifacts`。

## k=1，目标 extra/row=1

| 策略 | 实际extra/row | final recall | recall gain | prefix-zero高风险recall | selected future ASR | selected_n |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.95 | 0.795 | 0.010 | 0.049 | 0.629 | 300.0 |
| prefix_hazard_weighted_asr | 0.95 | 0.785 | 0.000 | 0.000 | 0.797 | 300.0 |
| prefix_alr | 0.95 | 0.795 | 0.009 | 0.048 | 0.758 | 300.0 |
| model_base_features | 0.95 | 0.795 | 0.009 | 0.051 | 0.767 | 300.0 |
| model_text_only | 0.95 | 0.793 | 0.008 | 0.036 | 0.793 | 300.0 |
| model_base_plus_text_residual | 0.95 | 0.794 | 0.008 | 0.041 | 0.782 | 300.0 |
| hybrid_hazard_text_q0.25 | 0.95 | 0.795 | 0.010 | 0.102 | 0.741 | 300.0 |
| hybrid_hazard_text_q0.5 | 0.95 | 0.804 | 0.019 | 0.188 | 0.677 | 300.0 |

## k=1，目标 extra/row=2

| 策略 | 实际extra/row | final recall | recall gain | prefix-zero高风险recall | selected future ASR | selected_n |
|---|---:|---:|---:|---:|---:|---:|
| random | 1.90 | 0.807 | 0.022 | 0.103 | 0.627 | 600.0 |
| prefix_hazard_weighted_asr | 1.90 | 0.786 | 0.001 | 0.001 | 0.776 | 600.0 |
| prefix_alr | 1.90 | 0.803 | 0.017 | 0.078 | 0.741 | 600.0 |
| model_base_features | 1.90 | 0.804 | 0.018 | 0.099 | 0.745 | 600.0 |
| model_text_only | 1.90 | 0.801 | 0.015 | 0.074 | 0.775 | 600.0 |
| model_base_plus_text_residual | 1.90 | 0.802 | 0.017 | 0.084 | 0.763 | 600.0 |
| hybrid_hazard_text_q0.25 | 1.90 | 0.804 | 0.019 | 0.188 | 0.721 | 600.0 |
| hybrid_hazard_text_q0.5 | 1.90 | 0.827 | 0.042 | 0.388 | 0.630 | 600.0 |

## k=1，目标 extra/row=3

| 策略 | 实际extra/row | final recall | recall gain | prefix-zero高风险recall | selected future ASR | selected_n |
|---|---:|---:|---:|---:|---:|---:|
| random | 2.85 | 0.818 | 0.032 | 0.145 | 0.620 | 900.0 |
| prefix_hazard_weighted_asr | 2.85 | 0.789 | 0.003 | 0.004 | 0.761 | 900.0 |
| prefix_alr | 2.85 | 0.813 | 0.028 | 0.133 | 0.731 | 900.0 |
| model_base_features | 2.85 | 0.813 | 0.028 | 0.156 | 0.739 | 900.0 |
| model_text_only | 2.85 | 0.809 | 0.024 | 0.113 | 0.763 | 900.0 |
| model_base_plus_text_residual | 2.85 | 0.811 | 0.026 | 0.126 | 0.753 | 900.0 |
| hybrid_hazard_text_q0.25 | 2.85 | 0.814 | 0.029 | 0.264 | 0.712 | 900.0 |
| hybrid_hazard_text_q0.5 | 2.85 | 0.841 | 0.055 | 0.487 | 0.632 | 900.0 |

## k=1，目标 extra/row=4

| 策略 | 实际extra/row | final recall | recall gain | prefix-zero高风险recall | selected future ASR | selected_n |
|---|---:|---:|---:|---:|---:|---:|
| random | 3.99 | 0.830 | 0.045 | 0.207 | 0.631 | 1260.0 |
| prefix_hazard_weighted_asr | 3.99 | 0.794 | 0.009 | 0.006 | 0.749 | 1260.0 |
| prefix_alr | 3.99 | 0.823 | 0.037 | 0.174 | 0.730 | 1260.0 |
| model_base_features | 3.99 | 0.824 | 0.039 | 0.209 | 0.734 | 1260.0 |
| model_text_only | 3.99 | 0.819 | 0.034 | 0.159 | 0.752 | 1260.0 |
| model_base_plus_text_residual | 3.99 | 0.822 | 0.037 | 0.183 | 0.744 | 1260.0 |
| hybrid_hazard_text_q0.25 | 3.99 | 0.831 | 0.046 | 0.391 | 0.690 | 1260.0 |
| hybrid_hazard_text_q0.5 | 3.99 | 0.860 | 0.074 | 0.611 | 0.623 | 1260.0 |

## k=2，目标 extra/row=1

| 策略 | 实际extra/row | final recall | recall gain | prefix-zero高风险recall | selected future ASR | selected_n |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.90 | 0.870 | 0.006 | 0.045 | 0.629 | 300.0 |
| prefix_hazard_weighted_asr | 0.90 | 0.864 | 0.000 | 0.000 | 0.824 | 300.0 |
| prefix_alr | 0.90 | 0.870 | 0.006 | 0.063 | 0.777 | 300.0 |
| model_base_features | 0.90 | 0.870 | 0.006 | 0.062 | 0.783 | 300.0 |
| model_text_only | 0.90 | 0.869 | 0.004 | 0.027 | 0.810 | 300.0 |
| model_base_plus_text_residual | 0.90 | 0.869 | 0.005 | 0.041 | 0.797 | 300.0 |
| hybrid_hazard_text_q0.25 | 0.90 | 0.872 | 0.008 | 0.177 | 0.736 | 300.0 |
| hybrid_hazard_text_q0.5 | 0.90 | 0.879 | 0.015 | 0.302 | 0.643 | 300.0 |

## k=2，目标 extra/row=2

| 策略 | 实际extra/row | final recall | recall gain | prefix-zero高风险recall | selected future ASR | selected_n |
|---|---:|---:|---:|---:|---:|---:|
| random | 1.98 | 0.879 | 0.015 | 0.107 | 0.625 | 660.0 |
| prefix_hazard_weighted_asr | 1.98 | 0.865 | 0.000 | 0.002 | 0.802 | 660.0 |
| prefix_alr | 1.98 | 0.876 | 0.011 | 0.088 | 0.760 | 660.0 |
| model_base_features | 1.98 | 0.877 | 0.012 | 0.115 | 0.765 | 660.0 |
| model_text_only | 1.98 | 0.875 | 0.010 | 0.065 | 0.787 | 660.0 |
| model_base_plus_text_residual | 1.98 | 0.875 | 0.011 | 0.073 | 0.775 | 660.0 |
| hybrid_hazard_text_q0.25 | 1.98 | 0.879 | 0.015 | 0.302 | 0.724 | 660.0 |
| hybrid_hazard_text_q0.5 | 1.98 | 0.896 | 0.031 | 0.554 | 0.611 | 660.0 |

## k=2，目标 extra/row=3

| 策略 | 实际extra/row | final recall | recall gain | prefix-zero高风险recall | selected future ASR | selected_n |
|---|---:|---:|---:|---:|---:|---:|
| random | 2.88 | 0.886 | 0.021 | 0.178 | 0.620 | 960.0 |
| prefix_hazard_weighted_asr | 2.88 | 0.865 | 0.001 | 0.000 | 0.786 | 960.0 |
| prefix_alr | 2.88 | 0.883 | 0.018 | 0.149 | 0.749 | 960.0 |
| model_base_features | 2.88 | 0.882 | 0.018 | 0.167 | 0.755 | 960.0 |
| model_text_only | 2.88 | 0.879 | 0.015 | 0.094 | 0.773 | 960.0 |
| model_base_plus_text_residual | 2.88 | 0.882 | 0.017 | 0.115 | 0.765 | 960.0 |
| hybrid_hazard_text_q0.25 | 2.88 | 0.891 | 0.026 | 0.486 | 0.687 | 960.0 |
| hybrid_hazard_text_q0.5 | 2.88 | 0.907 | 0.043 | 0.695 | 0.598 | 960.0 |

## k=2，目标 extra/row=4

| 策略 | 实际extra/row | final recall | recall gain | prefix-zero高风险recall | selected future ASR | selected_n |
|---|---:|---:|---:|---:|---:|---:|
| random | 3.96 | 0.894 | 0.030 | 0.219 | 0.631 | 1320.0 |
| prefix_hazard_weighted_asr | 3.96 | 0.866 | 0.002 | 0.007 | 0.770 | 1320.0 |
| prefix_alr | 3.96 | 0.887 | 0.023 | 0.160 | 0.742 | 1320.0 |
| model_base_features | 3.96 | 0.889 | 0.025 | 0.209 | 0.742 | 1320.0 |
| model_text_only | 3.96 | 0.886 | 0.022 | 0.120 | 0.759 | 1320.0 |
| model_base_plus_text_residual | 3.96 | 0.889 | 0.025 | 0.178 | 0.753 | 1320.0 |
| hybrid_hazard_text_q0.25 | 3.96 | 0.896 | 0.032 | 0.554 | 0.687 | 1320.0 |
| hybrid_hazard_text_q0.5 | 3.96 | 0.917 | 0.053 | 0.765 | 0.605 | 1320.0 |

## k=4，目标 extra/row=1

| 策略 | 实际extra/row | final recall | recall gain | prefix-zero高风险recall | selected future ASR | selected_n |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.96 | 0.923 | 0.004 | 0.061 | 0.627 | 360.0 |
| prefix_hazard_weighted_asr | 0.96 | 0.919 | 0.000 | 0.000 | 0.850 | 360.0 |
| prefix_alr | 0.96 | 0.923 | 0.004 | 0.087 | 0.786 | 360.0 |
| model_base_features | 0.96 | 0.923 | 0.004 | 0.048 | 0.785 | 360.0 |
| model_text_only | 0.96 | 0.922 | 0.003 | 0.040 | 0.813 | 360.0 |
| model_base_plus_text_residual | 0.96 | 0.923 | 0.004 | 0.026 | 0.800 | 360.0 |
| hybrid_hazard_text_q0.25 | 0.96 | 0.924 | 0.006 | 0.293 | 0.750 | 360.0 |
| hybrid_hazard_text_q0.5 | 0.96 | 0.931 | 0.013 | 0.509 | 0.579 | 360.0 |

## k=4，目标 extra/row=2

| 策略 | 实际extra/row | final recall | recall gain | prefix-zero高风险recall | selected future ASR | selected_n |
|---|---:|---:|---:|---:|---:|---:|
| random | 1.92 | 0.929 | 0.010 | 0.137 | 0.625 | 720.0 |
| prefix_hazard_weighted_asr | 1.92 | 0.919 | 0.000 | 0.000 | 0.819 | 720.0 |
| prefix_alr | 1.92 | 0.926 | 0.007 | 0.106 | 0.770 | 720.0 |
| model_base_features | 1.92 | 0.927 | 0.008 | 0.159 | 0.766 | 720.0 |
| model_text_only | 1.92 | 0.926 | 0.007 | 0.103 | 0.790 | 720.0 |
| model_base_plus_text_residual | 1.92 | 0.927 | 0.008 | 0.133 | 0.781 | 720.0 |
| hybrid_hazard_text_q0.25 | 1.92 | 0.931 | 0.013 | 0.509 | 0.698 | 720.0 |
| hybrid_hazard_text_q0.5 | 1.92 | 0.939 | 0.021 | 0.641 | 0.603 | 720.0 |

## k=4，目标 extra/row=3

| 策略 | 实际extra/row | final recall | recall gain | prefix-zero高风险recall | selected future ASR | selected_n |
|---|---:|---:|---:|---:|---:|---:|
| random | 2.88 | 0.932 | 0.014 | 0.203 | 0.622 | 1080.0 |
| prefix_hazard_weighted_asr | 2.88 | 0.919 | 0.000 | 0.000 | 0.799 | 1080.0 |
| prefix_alr | 2.88 | 0.931 | 0.012 | 0.179 | 0.752 | 1080.0 |
| model_base_features | 2.88 | 0.932 | 0.013 | 0.199 | 0.753 | 1080.0 |
| model_text_only | 2.88 | 0.929 | 0.010 | 0.169 | 0.773 | 1080.0 |
| model_base_plus_text_residual | 2.88 | 0.930 | 0.012 | 0.184 | 0.766 | 1080.0 |
| hybrid_hazard_text_q0.25 | 2.88 | 0.934 | 0.015 | 0.590 | 0.702 | 1080.0 |
| hybrid_hazard_text_q0.5 | 2.88 | 0.945 | 0.026 | 0.676 | 0.619 | 1080.0 |

## k=4，目标 extra/row=4

| 策略 | 实际extra/row | final recall | recall gain | prefix-zero高风险recall | selected future ASR | selected_n |
|---|---:|---:|---:|---:|---:|---:|
| random | 4.00 | 0.939 | 0.020 | 0.343 | 0.631 | 1500.0 |
| prefix_hazard_weighted_asr | 4.00 | 0.920 | 0.001 | 0.000 | 0.780 | 1500.0 |
| prefix_alr | 4.00 | 0.935 | 0.016 | 0.236 | 0.745 | 1500.0 |
| model_base_features | 4.00 | 0.936 | 0.018 | 0.261 | 0.742 | 1500.0 |
| model_text_only | 4.00 | 0.934 | 0.015 | 0.204 | 0.760 | 1500.0 |
| model_base_plus_text_residual | 4.00 | 0.935 | 0.017 | 0.213 | 0.753 | 1500.0 |
| hybrid_hazard_text_q0.25 | 4.00 | 0.940 | 0.021 | 0.641 | 0.689 | 1500.0 |
| hybrid_hazard_text_q0.5 | 4.00 | 0.951 | 0.032 | 0.724 | 0.634 | 1500.0 |

