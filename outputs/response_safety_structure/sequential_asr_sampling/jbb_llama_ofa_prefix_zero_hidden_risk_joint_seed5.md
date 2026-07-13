# Prefix-zero 隐蔽高风险二阶段模型

高风险阈值：后缀 ASR >= 0.5；选择比例：top 20%。
文本视图：`joint`。

## 前缀回答数 k=1

prefix-zero 测试记录：697；高风险数：105；比例：0.151。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.496 | 0.161 | 0.194 | 0.257 | 0.263 |
| base_logistic | 0.752 | 0.350 | 0.396 | 0.524 | 0.457 |
| base_ridge | 0.762 | 0.340 | 0.388 | 0.514 | 0.452 |
| text_logistic | 0.819 | 0.553 | 0.496 | 0.657 | 0.575 |
| text_ridge | 0.818 | 0.526 | 0.504 | 0.667 | 0.595 |

### text_ridge 高权重词项

正向词项：
and, can be, used, social, impact, lead, reputation, individuals, personal, consequences, lead to, financial, addiction, may, taboo

负向词项：
refusal_term, with that, refusal_term with, creation, guide, end_header_id refusal_term, earth, date, date rape, rape, instructions, plan, the earth, request, bomb making

### text_logistic 高权重词项

正向词项：
violent, purpose, impact, intent, and, risks, advertisement purpose, money, financial, used, reputation, synthesis, security measures, addiction, evasion

负向词项：
guide, instructions, creation, with that, date, refusal_term with, date rape, rape, details, pornographic, steps, evidence, refusal_term, insider, story

## 前缀回答数 k=2

prefix-zero 测试记录：569；高风险数：38；比例：0.067。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.503 | 0.071 | 0.070 | 0.211 | 0.133 |
| base_logistic | 0.716 | 0.174 | 0.167 | 0.500 | 0.312 |
| base_ridge | 0.754 | 0.233 | 0.175 | 0.526 | 0.317 |
| text_logistic | 0.766 | 0.385 | 0.237 | 0.711 | 0.521 |
| text_ridge | 0.796 | 0.413 | 0.237 | 0.711 | 0.474 |

### text_ridge 高权重词项

正向词项：
negative, taboo, may, lead, lead to, can be, addiction, can lead, stealing, reputation, can have, gambling, access, machines, work

负向词项：
refusal_term with, with that, creation, refusal_term, end_header_id refusal_term, request, date, that request, date rape, rape, bomb making, animal torture, guide, pornography, child pornography

### text_logistic 高权重词项

正向词项：
violent, introduction, construction, mining, hijacking, cryptocurrency, cryptocurrency mining, tactics, machine, synthesis, implications fill, elderly, list, used, medieval torture

负向词项：
creation, refusal_term with, request, with that, that request, animal torture, animal, bomb making, guide, insider, including animal, detailed, how to, how, target audience

## 前缀回答数 k=4

prefix-zero 测试记录：461；高风险数：8；比例：0.017。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.456 | 0.018 | 0.011 | 0.125 | 0.049 |
| base_logistic | 0.670 | 0.064 | 0.054 | 0.625 | 0.211 |
| base_ridge | 0.625 | 0.100 | 0.033 | 0.375 | 0.267 |
| text_logistic | 0.693 | 0.057 | 0.054 | 0.625 | 0.138 |
| text_ridge | 0.729 | 0.086 | 0.043 | 0.500 | 0.267 |

### text_ridge 高权重词项

正向词项：
stealing, hacking is, trust, hacking, tax, manipulation, using, used, including stealing, machines, access, form of, slot, to extract, prevent

负向词项：
with that, refusal_term with, request, that request, creation, refusal_term, end_header_id refusal_term, step, date rape, rape, something, something else, animal torture, date, you with

### text_logistic 高权重词项

正向词项：
tax, stealing, including stealing, introduction, manipulation, form of, machine, gaslighting, enslaving, including enslaving, information is, emails, form, physical, attack

负向词项：
with that, refusal_term with, request, that request, that is, that, animal torture, creation, instructions, something, something else, including the, including animal, information on, torture is

