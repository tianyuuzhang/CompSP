# Prefix-zero 隐蔽高风险二阶段模型

高风险阈值：后缀 ASR >= 0.5；选择比例：top 20%。
文本视图：`joint`。

## 前缀回答数 k=1

prefix-zero 测试记录：2219；高风险数：319；比例：0.144。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.508 | 0.147 | 0.146 | 0.204 | 0.254 |
| base_logistic | 0.750 | 0.299 | 0.306 | 0.426 | 0.451 |
| base_ridge | 0.749 | 0.297 | 0.304 | 0.423 | 0.453 |
| text_logistic | 0.786 | 0.430 | 0.360 | 0.502 | 0.435 |
| text_ridge | 0.773 | 0.404 | 0.358 | 0.498 | 0.435 |

### text_ridge 高权重词项

正向词项：
article, of, kiwi, consequences, post, it, end_header_id user, media, business, individuals, and, social, social media, personal, addiction

负向词项：
end_header_id refusal_term, refusal_term, with that, request, refusal_term with, that request, flat, team, evidence, football, teams, earth, the earth, creation, illuminati

### text_logistic 高权重词项

正向词项：
purpose, risks, security measures, article, of, media, and, it, network, draft instructions, distribute, so, business, words, taboo

负向词项：
end_header_id refusal_term, with that, refusal_term, refusal_term with, request, that request, flat, general, illuminati, idea, the illuminati, date rape, rape, pornographic, evidence

## 前缀回答数 k=2

prefix-zero 测试记录：1846；高风险数：101；比例：0.055。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.510 | 0.060 | 0.054 | 0.198 | 0.108 |
| base_logistic | 0.735 | 0.122 | 0.136 | 0.495 | 0.228 |
| base_ridge | 0.730 | 0.113 | 0.119 | 0.436 | 0.224 |
| text_logistic | 0.769 | 0.186 | 0.157 | 0.574 | 0.284 |
| text_ridge | 0.771 | 0.209 | 0.157 | 0.574 | 0.278 |

### text_ridge 高权重词项

正向词项：
addiction, end_header_id user, article, taboo, post, kiwi, it, post question, importance, social, consequences, post that, individuals, gambling, more

负向词项：
end_header_id refusal_term, refusal_term, request, with that, refusal_term with, that request, flat, team, football, bomb making, creation, here is, date, sure here, your

### text_logistic 高权重词项

正向词项：
introduction, so, gambling, consequences, mining, hijacking, techniques, cryptocurrency, cryptocurrency mining, fear, topic is, those, taboo, after, people

负向词项：
with that, refusal_term with, that request, request, refusal_term, end_header_id refusal_term, creation, bomb making, planet, on how, who is, flat, animal torture, global events, illuminati

## 前缀回答数 k=4

prefix-zero 测试记录：1568；高风险数：16；比例：0.010。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.396 | 0.008 | 0.000 | 0.000 | 0.021 |
| base_logistic | 0.692 | 0.031 | 0.025 | 0.500 | 0.091 |
| base_ridge | 0.738 | 0.044 | 0.029 | 0.562 | 0.105 |
| text_logistic | 0.835 | 0.127 | 0.032 | 0.625 | 0.190 |
| text_ridge | 0.834 | 0.186 | 0.041 | 0.812 | 0.261 |

### text_ridge 高权重词项

正向词项：
end_header_id user, kiwi, article, issue, error, code, an article, more, manipulation, q1_generation_failed_a0_error, encountered, tools, error code, impact, are kiwi

负向词项：
end_header_id refusal_term, request, refusal_term, with that, that request, refusal_term with, request 回答分隔, you are, here is, flat, sure, sure here, your, football, with sure

### text_logistic 高权重词项

正向词项：
tax, introduction, manipulation, information is, documents, attack, enslaving, individuals, form of, theme, emails, form, techniques, engineering, cold

负向词项：
refusal_term with, with that, request, that request, that, end_header_id refusal_term, refusal_term, request 回答分隔, including the, earth, creation, real, your, that is, instructions

