# 自然前缀对白盒 H_proxy 的条件增量实验

当前只覆盖 legacy 20 题 PAIR/DrAttack，属于 smoke/探索，不作为最终机制证据。

| k | 模型 | mean layer Spearman | last layer Spearman | vector cosine | R2 |
|---:|---|---:|---:|---:|---:|
| 1 | count_q1 | 0.703 | 0.447 | 0.997 | 0.520 |
| 1 | count_q1_response | 0.715 | 0.460 | 0.997 | 0.564 |
| 1 | delta_response_minus_base | 0.013 | 0.014 | 0.000 | 0.044 |
| 2 | count_q1 | 0.701 | 0.440 | 0.997 | 0.514 |
| 2 | count_q1_response | 0.717 | 0.466 | 0.997 | 0.570 |
| 2 | delta_response_minus_base | 0.016 | 0.026 | 0.000 | 0.056 |
| 4 | count_q1 | 0.702 | 0.445 | 0.997 | 0.496 |
| 4 | count_q1_response | 0.720 | 0.477 | 0.997 | 0.565 |
| 4 | delta_response_minus_base | 0.019 | 0.032 | 0.001 | 0.069 |
