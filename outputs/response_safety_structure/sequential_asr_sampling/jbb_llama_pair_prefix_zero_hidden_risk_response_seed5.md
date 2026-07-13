# Prefix-zero 隐蔽高风险二阶段模型

高风险阈值：后缀 ASR >= 0.5；选择比例：top 20%。
文本视图：`response`。

## 前缀回答数 k=1

prefix-zero 测试记录：619；高风险数：108；比例：0.174。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.518 | 0.170 | 0.145 | 0.167 | 0.321 |
| base_logistic | 0.773 | 0.438 | 0.379 | 0.435 | 0.513 |
| base_ridge | 0.763 | 0.431 | 0.379 | 0.435 | 0.501 |
| text_logistic | 0.789 | 0.483 | 0.435 | 0.500 | 0.547 |
| text_ridge | 0.785 | 0.467 | 0.435 | 0.500 | 0.544 |

### text_ridge 高权重词项

正向词项：
she, relationships, step, only, we, love, my, who, partner, series, her, local, connection, breach, series of

负向词项：
end_header_id refusal_term, refusal_term, earth, the earth, flat, refusal_term provide, game, that request, teams, request, team, the game, football, flat earth, evidence

### text_logistic 高权重词项

正向词项：
only, breach, harm, step, relationships, made, love, partner, message q1_generation_failed_a0_error, may, we, so, materials, empire, of our

负向词项：
earth, end_header_id refusal_term, the earth, specific, refusal_term, that request, will help, team, this will, flat, general, refusal_term provide, request, the game, any

## 前缀回答数 k=2

prefix-zero 测试记录：501；高风险数：33；比例：0.066。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.512 | 0.067 | 0.050 | 0.152 | 0.146 |
| base_logistic | 0.609 | 0.110 | 0.150 | 0.455 | 0.257 |
| base_ridge | 0.480 | 0.068 | 0.080 | 0.242 | 0.142 |
| text_logistic | 0.780 | 0.237 | 0.220 | 0.667 | 0.382 |
| text_ridge | 0.742 | 0.174 | 0.230 | 0.697 | 0.361 |

### text_ridge 高权重词项

正向词项：
error, only, system, love, but, materials, had, who, was, where, we, access to, use, people, our

负向词项：
end_header_id refusal_term, refusal_term, earth, the earth, flat, can help, you with, help you, else, refusal_term provide, is there, else can, teams, there anything, that request

### text_logistic 高权重词项

正向词项：
ideology, fear, materials, only, so can, childhood, people to, so, love, referencing an, error the, large, cultural, long, specific system

负向词项：
that request, earth, end_header_id refusal_term, request, refusal_term with, with that, refusal_term, the earth, team, problem, game, help you, request is, including, flat

## 前缀回答数 k=4

prefix-zero 测试记录：409；高风险数：3；比例：0.007。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.615 | 0.017 | 0.012 | 0.333 | 0.047 |
| base_logistic | 0.566 | 0.020 | 0.012 | 0.333 | 0.071 |
| base_ridge | 0.370 | 0.008 | 0.000 | 0.000 | 0.021 |
| text_logistic | 0.960 | 0.394 | 0.037 | 1.000 | 0.500 |
| text_ridge | 0.993 | 0.443 | 0.037 | 1.000 | 0.600 |

### text_ridge 高权重词项

正向词项：
error, code, encountered, error code, q1_generation_failed_a0_error, this error, error message, provide more, more context, an error, context or, about where, ve encountered, encountered this, end_header_id it

负向词项：
end_header_id refusal_term, refusal_term, you with, earth, is there, refusal_term provide, else, else can, can help, the earth, flat, there anything, with 回答分隔, anything else, request

### text_logistic 高权重词项

正向词项：
code, error code, accurate response, on what, can you, you troubleshoot, or details, troubleshoot, cold, troubleshoot or, and provide, large, the cold, access to, alex

负向词项：
end_header_id refusal_term, refusal_term, you with, else can, request, else, is there, can help, refusal_term provide, there, with 回答分隔, do my, my best, best to, ll do

