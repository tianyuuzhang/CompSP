# Prefix-zero 隐蔽高风险二阶段模型

高风险阈值：后缀 ASR >= 0.5；选择比例：top 20%。
文本视图：`joint`。

## 前缀回答数 k=1

prefix-zero 测试记录：619；高风险数：108；比例：0.174。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.518 | 0.170 | 0.145 | 0.167 | 0.321 |
| base_logistic | 0.773 | 0.438 | 0.379 | 0.435 | 0.513 |
| base_ridge | 0.763 | 0.431 | 0.379 | 0.435 | 0.501 |
| text_logistic | 0.805 | 0.490 | 0.452 | 0.519 | 0.562 |
| text_ridge | 0.790 | 0.444 | 0.444 | 0.509 | 0.554 |

### text_ridge 高权重词项

正向词项：
she, we, my, relationships, her, in the, partner, series, step, through, love, series of, it, our, people

负向词项：
end_header_id refusal_term, earth, refusal_term, request, that request, the earth, flat, is there, you with, else can, help you, is, your, on how, football

### text_logistic 高权重词项

正向词项：
we, partner, vulnerabilities, so can, only, exploited, she, so, measures, made, message q1_generation_failed_a0_error, step, in the, not, relationships

负向词项：
end_header_id refusal_term, refusal_term, that request, earth, specific, request, the earth, help, you with, will help, flat, help you, general, the model, with that

## 前缀回答数 k=2

prefix-zero 测试记录：501；高风险数：33；比例：0.066。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.512 | 0.067 | 0.050 | 0.152 | 0.146 |
| base_logistic | 0.609 | 0.110 | 0.150 | 0.455 | 0.257 |
| base_ridge | 0.480 | 0.068 | 0.080 | 0.242 | 0.142 |
| text_logistic | 0.783 | 0.240 | 0.230 | 0.697 | 0.367 |
| text_ridge | 0.744 | 0.176 | 0.220 | 0.667 | 0.349 |

### text_ridge 高权重词项

正向词项：
error, system, q1_generation_failed_a0_error, we, was, only, had, use, love, my, but, it, long, our, more

负向词项：
end_header_id refusal_term, request, that request, refusal_term, you with, earth, can help, is there, else can, else, help you, there anything, the earth, flat, anything else

### text_logistic 高权重词项

正向词项：
so can, so, only, after, childhood, fear, referencing an, love, poem, ideology, large, long, people to, specific system, materials

负向词项：
end_header_id refusal_term, that request, request, help you, refusal_term, earth, game, you with, the earth, on how, problem, help, with that, they, flat

## 前缀回答数 k=4

prefix-zero 测试记录：409；高风险数：3；比例：0.007。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.615 | 0.017 | 0.012 | 0.333 | 0.047 |
| base_logistic | 0.566 | 0.020 | 0.012 | 0.333 | 0.071 |
| base_ridge | 0.370 | 0.008 | 0.000 | 0.000 | 0.021 |
| text_logistic | 0.964 | 0.236 | 0.037 | 1.000 | 0.400 |
| text_ridge | 0.991 | 0.387 | 0.037 | 1.000 | 0.545 |

### text_ridge 高权重词项

正向词项：
error, encountered, q1_generation_failed_a0_error, code, error code, this error, error message, provide more, more context, context or, an error, about where, ve encountered, encountered this, end_header_id it

负向词项：
end_header_id refusal_term, you with, refusal_term, is there, else can, request, else, that request, with 回答分隔, can help, there anything, refusal_term provide, anything else, earth, anything

### text_logistic 高权重词项

正向词项：
code, error code, accurate response, on what, you troubleshoot, troubleshoot, or details, cold, troubleshoot or, and provide, large, the official, response 回答分隔, access to, issue and

负向词项：
end_header_id refusal_term, refusal_term, you with, else can, request, is there, else, can help, refusal_term provide, with 回答分隔, there anything, there, do my, my best, that request

