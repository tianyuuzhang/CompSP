# Prefix-zero 隐蔽高风险二阶段模型

高风险阈值：后缀 ASR >= 0.5；选择比例：top 20%。
文本视图：`response`。
回答归属打乱：`within_question_attack`。

## 前缀回答数 k=1

prefix-zero 测试记录：2219；高风险数：319；比例：0.144。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.508 | 0.147 | 0.146 | 0.204 | 0.254 |
| base_logistic | 0.750 | 0.299 | 0.306 | 0.426 | 0.451 |
| base_ridge | 0.749 | 0.297 | 0.304 | 0.423 | 0.453 |
| text_logistic | 0.649 | 0.234 | 0.252 | 0.351 | 0.340 |
| text_ridge | 0.640 | 0.215 | 0.248 | 0.345 | 0.326 |

### text_ridge 高权重词项

正向词项：
article, post, watermelon, kiwi, blog post, addiction, write, consequences, blog, their, pineapple, individuals, an article, to write, and

负向词项：
flat, refusal_term, end_header_id refusal_term, ransomware, bomb, team, refusal_term provide, animal torture, email, game, planet, shape, animal, earth, the earth

### text_logistic 高权重词项

正向词项：
gambling, article, post, watermelon, with something, materials, pineapple, individuals, has, breach, their, behavior, consequences, cheating is, income

负向词项：
bomb, flat, substances, refusal_term, different, pornography, animal, including hacking, planet, child pornography, that the, shape, date rape, rape, illuminati

## 前缀回答数 k=2

prefix-zero 测试记录：1846；高风险数：101；比例：0.055。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.510 | 0.060 | 0.054 | 0.198 | 0.108 |
| base_logistic | 0.735 | 0.122 | 0.136 | 0.495 | 0.228 |
| base_ridge | 0.730 | 0.113 | 0.119 | 0.436 | 0.224 |
| text_logistic | 0.627 | 0.093 | 0.070 | 0.257 | 0.143 |
| text_ridge | 0.637 | 0.087 | 0.087 | 0.317 | 0.147 |

### text_ridge 高权重词项

正向词项：
post, article, kiwi, consequences, addiction, mango, an article, blog post, blog, post that, write an, an, guide on, pineapple, official

负向词项：
refusal_term, flat, animal torture, ransomware, end_header_id refusal_term, animal, including animal, date, promotes, that promotes, date rape, rape, write sexist, pedophilia, bomb

### text_logistic 高权重词项

正向词项：
promote, that is, tax, stalking, post, article, bias, assistant refusal_term, cards, an article, pineapple, design, addiction, document, database

负向词项：
ransomware, provide guidance, holocaust, animal, trading is, cannibalism, animal torture, suicide, date, response, the holocaust, flat, cannibalism is, debit card, pornography is

## 前缀回答数 k=4

prefix-zero 测试记录：1568；高风险数：16；比例：0.010。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.396 | 0.008 | 0.000 | 0.000 | 0.021 |
| base_logistic | 0.692 | 0.031 | 0.025 | 0.500 | 0.091 |
| base_ridge | 0.738 | 0.044 | 0.029 | 0.562 | 0.105 |
| text_logistic | 0.717 | 0.070 | 0.022 | 0.438 | 0.160 |
| text_ridge | 0.714 | 0.049 | 0.022 | 0.438 | 0.108 |

### text_ridge 高权重词项

正向词项：
article, an article, kiwi, blog, blog post, were, medieval, post, cheat, an, context, as, exam, behavior, write an

负向词项：
refusal_term, end_header_id refusal_term, flat, animal torture, child, bomb, including animal, write sexist, harm, refusal_term provide, rape, date rape, with that, that promotes, animal

### text_logistic 高权重词项

正向词项：
information is, requests, hate, medieval, discussing, effects, bias, atm hacking, request 回答分隔, atm, erotic, an article, tax, medieval torture, describe medieval

负向词项：
request is, that 回答分隔, refusal_term you, including hacking, including how, animal, pornography is, counterfeit, animal torture, hazard_term is, provide guidance, including animal, that promotes, promotes, provide response

