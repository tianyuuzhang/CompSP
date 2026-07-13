# Prefix-zero 隐蔽高风险二阶段模型

高风险阈值：后缀 ASR >= 0.5；选择比例：top 20%。
文本视图：`q1`。

## 前缀回答数 k=1

prefix-zero 测试记录：2219；高风险数：319；比例：0.144。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.508 | 0.147 | 0.146 | 0.204 | 0.254 |
| base_logistic | 0.750 | 0.299 | 0.306 | 0.426 | 0.451 |
| base_ridge | 0.749 | 0.297 | 0.304 | 0.423 | 0.453 |
| text_logistic | 0.621 | 0.238 | 0.248 | 0.345 | 0.297 |
| text_ridge | 0.572 | 0.216 | 0.241 | 0.335 | 0.300 |

### text_ridge 高权重词项

正向词项：
article, impact, post, addiction, gambling, kiwi, post question, security measures, slot, blog, write blog, slot machines, ethical, financial, plagiarism

负向词项：
guide, evidence, creation, earth, plan, instructions, animal torture, flat, insider, football, child, date rape, date, rape, insider tips

### text_logistic 高权重词项

正向词项：
article, impact, addiction, gambling, post, network, security measures, financial, slot, consequences, plagiarism, discrimination, unconventional, detail, business network

负向词项：
theory, evidence, earth, guide, flat, insider, pornographic, creation, hoax, date rape, date, rape, promoting, shape, instructions

## 前缀回答数 k=2

prefix-zero 测试记录：1846；高风险数：101；比例：0.055。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.510 | 0.060 | 0.054 | 0.198 | 0.108 |
| base_logistic | 0.735 | 0.122 | 0.136 | 0.495 | 0.228 |
| base_ridge | 0.730 | 0.113 | 0.119 | 0.436 | 0.224 |
| text_logistic | 0.612 | 0.127 | 0.079 | 0.287 | 0.220 |
| text_ridge | 0.559 | 0.061 | 0.065 | 0.238 | 0.130 |

### text_ridge 高权重词项

正向词项：
article, addiction, gambling, impact, kiwi, post, slot, machines, write blog, story, slot machines, negative, gender, post question, refusal_term refusal_term

负向词项：
creation, instructions, guide, earth, plan, animal torture, football, insider, steps, flat, insider tips, animal, child, target audience, date

### text_logistic 高权重词项

正向词项：
gambling, article, addiction, machine, relationship, character, opinion, slot, romantic, discrimination, impact, plagiarism detection, medieval, gender, post

负向词项：
creation, insider, earth, instructions, flat, theory, provide, guide, steps, target audience, control, pornographic, avoiding detection, shape, they

## 前缀回答数 k=4

prefix-zero 测试记录：1568；高风险数：16；比例：0.010。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.396 | 0.008 | 0.000 | 0.000 | 0.021 |
| base_logistic | 0.692 | 0.031 | 0.025 | 0.500 | 0.091 |
| base_ridge | 0.738 | 0.044 | 0.029 | 0.562 | 0.105 |
| text_logistic | 0.444 | 0.009 | 0.003 | 0.062 | 0.022 |
| text_ridge | 0.469 | 0.009 | 0.003 | 0.062 | 0.023 |

### text_ridge 高权重词项

正向词项：
refusal_term refusal_term, q1_generation_failed_a0_error refusal_term, q1_generation_failed_a0_error, with that, article, refusal_term with, machines, kiwi, refusal_term, medieval, scientific evidence, autism, slot machines, slot, impact

负向词项：
creation, instructions, plan, earth, animal torture, campaign, football, making, guide, insider tips, animal, step, insider, flat, date

### text_logistic 高权重词项

正向词项：
scientific evidence, slot machines, environment, slot, article, medieval, introduction, tragic, machines, effects on, blog, refusal_term refusal_term, challenges, struggling, kiwi

负向词项：
instructions, earth, legal, world, risks, step, insider, control, creation, analysis, guide, influence, theory, promoting, creative

