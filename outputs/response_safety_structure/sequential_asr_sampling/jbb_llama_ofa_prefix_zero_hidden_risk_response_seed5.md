# Prefix-zero 隐蔽高风险二阶段模型

高风险阈值：后缀 ASR >= 0.5；选择比例：top 20%。
文本视图：`response`。

## 前缀回答数 k=1

prefix-zero 测试记录：697；高风险数：105；比例：0.151。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.496 | 0.161 | 0.194 | 0.257 | 0.263 |
| base_logistic | 0.752 | 0.350 | 0.396 | 0.524 | 0.457 |
| base_ridge | 0.762 | 0.340 | 0.388 | 0.514 | 0.452 |
| text_logistic | 0.797 | 0.535 | 0.525 | 0.695 | 0.611 |
| text_ridge | 0.809 | 0.519 | 0.540 | 0.714 | 0.623 |

### text_ridge 高权重词项

正向词项：
consequences, access, social, genocide, individuals, personal, access to, can be, financial, lead, unauthorized, reputation, addiction, behavior, unauthorized access

负向词项：
refusal_term, earth, date, end_header_id refusal_term, with that, refusal_term with, date rape, rape, animal, the earth, team, illuminati, animal torture, that, pornography

### text_logistic 高权重词项

正向词项：
violent, including unauthorized, unauthorized access, content is, unauthorized, money, request is, access, financial, on human, and, form of, reputation, attack, genocide

负向词项：
date, date rape, rape, on how, including the, including animal, pornography is, how to, animal, illuminati, theories, earth, how, pornographic, on creating

## 前缀回答数 k=2

prefix-zero 测试记录：569；高风险数：38；比例：0.067。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.503 | 0.071 | 0.070 | 0.211 | 0.133 |
| base_logistic | 0.716 | 0.174 | 0.167 | 0.500 | 0.312 |
| base_ridge | 0.754 | 0.233 | 0.175 | 0.526 | 0.317 |
| text_logistic | 0.768 | 0.353 | 0.228 | 0.684 | 0.489 |
| text_ridge | 0.775 | 0.375 | 0.237 | 0.711 | 0.458 |

### text_ridge 高权重词项

正向词项：
consequences, negative, content is, addiction, lead, lead to, may, can lead, behavior, can be, tax, reputation, explicit, content, machines

负向词项：
refusal_term with, with that, refusal_term, date, animal, animal torture, pornography is, end_header_id refusal_term, date rape, rape, including animal, pornography, child pornography, request, earth

### text_logistic 高权重词项

正向词项：
violent, cryptocurrency, cryptocurrency mining, mining, content is, topic is, behavior, eliminating, network, including cryptocurrency, elderly, people is, bomb is, including unauthorized, enslavement

负向词项：
including animal, animal, animal torture, refusal_term with, on how, pornography is, cannibalism, with that, how to, cannibalism is, how, request, that request, including hacking, insider

## 前缀回答数 k=4

prefix-zero 测试记录：461；高风险数：8；比例：0.017。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.456 | 0.018 | 0.011 | 0.125 | 0.049 |
| base_logistic | 0.670 | 0.064 | 0.054 | 0.625 | 0.211 |
| base_ridge | 0.625 | 0.100 | 0.033 | 0.375 | 0.267 |
| text_logistic | 0.698 | 0.081 | 0.054 | 0.625 | 0.182 |
| text_ridge | 0.730 | 0.115 | 0.054 | 0.625 | 0.308 |

### text_ridge 高权重词项

正向词项：
hacking is, hacking, stealing, trust, including stealing, colleague, using, used, consequences, victims, machines, techniques, manipulation, form of, extract

负向词项：
with that, refusal_term with, request, that request, rape, date rape, refusal_term, animal, something, pornography is, something else, end_header_id refusal_term, animal torture, date, torture is

### text_logistic 高权重词项

正向词项：
stealing, form of, including enslaving, including stealing, enslaving, manipulation, in any, form, atm, information is, erotic, topic is, including suicide, erotic content, content is

负向词项：
with that, refusal_term with, something, something else, that is, including the, request, that request, that, there something, information on, animal, animal torture, torture is, including animal

