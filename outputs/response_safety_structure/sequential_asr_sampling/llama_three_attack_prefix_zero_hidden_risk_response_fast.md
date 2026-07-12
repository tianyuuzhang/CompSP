# Prefix-zero 隐蔽高风险二阶段模型

高风险阈值：后缀 ASR >= 0.5；选择比例：top 20%。
文本视图：`response`。

## 前缀回答数 k=1

prefix-zero 测试记录：2219；高风险数：319；比例：0.144。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.508 | 0.147 | 0.146 | 0.204 | 0.254 |
| base_logistic | 0.750 | 0.299 | 0.306 | 0.426 | 0.451 |
| base_ridge | 0.749 | 0.297 | 0.304 | 0.423 | 0.453 |
| text_logistic | 0.781 | 0.402 | 0.354 | 0.492 | 0.457 |
| text_ridge | 0.787 | 0.414 | 0.390 | 0.542 | 0.467 |

### text_ridge 高权重词项

正向词项：
social, and, post, consequences, social media, who, step, media, individuals, of, article, pineapple, business, their, addiction

负向词项：
refusal_term, end_header_id refusal_term, flat, evidence, provide, refusal_term provide, planet, team, earth, illuminati, shape, the earth, the planet, the illuminati, game

### text_logistic 高权重词项

正向词项：
and, network, money, post, on human, social, pineapple, so, database, behavior, viewing, to, tax, life, draft

负向词项：
flat, general, illuminati, the illuminati, refusal_term, team, mango, planet, evidence, global events, specific, date rape, rape, refusal_term write, shape

## 前缀回答数 k=2

prefix-zero 测试记录：1846；高风险数：101；比例：0.055。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.510 | 0.060 | 0.054 | 0.198 | 0.108 |
| base_logistic | 0.735 | 0.122 | 0.136 | 0.495 | 0.228 |
| base_ridge | 0.730 | 0.113 | 0.119 | 0.436 | 0.224 |
| text_logistic | 0.777 | 0.177 | 0.168 | 0.614 | 0.265 |
| text_ridge | 0.785 | 0.195 | 0.168 | 0.614 | 0.275 |

### text_ridge 高权重词项

正向词项：
post, consequences, social, addiction, it, individuals, using, blog, kiwi, erotic, by, movie, behavior, gambling, consequences of

负向词项：
refusal_term, end_header_id refusal_term, flat, refusal_term write, planet, refusal_term provide, earth, the planet, evidence, that promotes, football, team, date, assistant refusal_term, teams

### text_logistic 高权重词项

正向词项：
gambling, materials, cryptocurrency, cryptocurrency mining, fear, with 回答分隔, mining, network, and using, so, and, individuals, behavior, ideology, eliminating

负向词项：
request is, ransomware, refusal_term with, on how, including animal, including hacking, with that, planet, pornography is, animal torture, cannibalism, flat, animal, creation of, global events

## 前缀回答数 k=4

prefix-zero 测试记录：1568；高风险数：16；比例：0.010。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.396 | 0.008 | 0.000 | 0.000 | 0.021 |
| base_logistic | 0.692 | 0.031 | 0.025 | 0.500 | 0.091 |
| base_ridge | 0.738 | 0.044 | 0.029 | 0.562 | 0.105 |
| text_logistic | 0.790 | 0.120 | 0.035 | 0.688 | 0.190 |
| text_ridge | 0.842 | 0.141 | 0.041 | 0.812 | 0.200 |

### text_ridge 高权重词项

正向词项：
code, error, issue, kiwi, it, manipulation, context, article, hacking is, encountered, error code, using, is are, an article, more

负向词项：
refusal_term, refusal_term write, flat, end_header_id refusal_term, request, refusal_term provide, with that, that promotes, refusal_term with, assistant refusal_term, football, promotes, teams, that request, evidence

### text_logistic 高权重词项

正向词项：
information is, enslaving, tax, refusal_term you, individuals, manipulation, atm, form of, that 回答分隔, form, deceive, including how, including suicide, content is, fulfill that

负向词项：
refusal_term with, that, ransomware, including the, with that, on how, that promotes, that is, promotes, refusal_term provide, content that, earth, refusal_term write, flat, real

