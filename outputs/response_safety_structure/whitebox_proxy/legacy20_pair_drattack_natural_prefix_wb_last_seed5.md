# 自然前缀对白盒 H_proxy 的条件增量实验

当前只覆盖 legacy 20 题 PAIR/DrAttack，属于 smoke/探索，不作为最终机制证据。

| k | 模型 | mean layer Spearman | last layer Spearman | vector cosine | R2 |
|---:|---|---:|---:|---:|---:|
| 1 | count_q1 | 0.732 | 0.678 | 0.982 | 0.663 |
| 1 | count_q1_response | 0.736 | 0.687 | 0.983 | 0.685 |
| 1 | delta_response_minus_base | 0.004 | 0.009 | 0.000 | 0.022 |
| 2 | count_q1 | 0.740 | 0.690 | 0.983 | 0.681 |
| 2 | count_q1_response | 0.748 | 0.711 | 0.984 | 0.707 |
| 2 | delta_response_minus_base | 0.009 | 0.020 | 0.000 | 0.026 |
| 4 | count_q1 | 0.743 | 0.691 | 0.983 | 0.682 |
| 4 | count_q1_response | 0.755 | 0.717 | 0.983 | 0.713 |
| 4 | delta_response_minus_base | 0.012 | 0.026 | 0.001 | 0.030 |
