# Prefix-zero 隐蔽高风险二阶段模型

高风险阈值：后缀 ASR >= 0.5；选择比例：top 20%。
文本视图：`q1`。

## 前缀回答数 k=1

prefix-zero 测试记录：697；高风险数：105；比例：0.151。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.496 | 0.161 | 0.194 | 0.257 | 0.263 |
| base_logistic | 0.752 | 0.350 | 0.396 | 0.524 | 0.457 |
| base_ridge | 0.762 | 0.340 | 0.388 | 0.514 | 0.452 |
| text_logistic | 0.727 | 0.312 | 0.338 | 0.448 | 0.409 |
| text_ridge | 0.708 | 0.275 | 0.281 | 0.371 | 0.382 |

### text_ridge 高权重词项

正向词项：
impact, addiction, gambling, financial, ethical, slot, machines, security measures, slot machines, article, consequences, denial, plagiarism, plagiarism detection, security

负向词项：
instructions, guide, plan, earth, creation, steps, animal torture, to, insider, date, rape, date rape, animal, rape drug, child

### text_logistic 高权重词项

正向词项：
impact, financial, addiction, consequences, gambling, slot, security measures, ethical, machines, purpose, article, slot machines, plagiarism, encryption, definition

负向词项：
instructions, guide, to, earth, insider, steps, evidence, control, pornographic, hoax, theory, establishing, betting, creation, date

## 前缀回答数 k=2

prefix-zero 测试记录：569；高风险数：38；比例：0.067。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.503 | 0.071 | 0.070 | 0.211 | 0.133 |
| base_logistic | 0.716 | 0.174 | 0.167 | 0.500 | 0.312 |
| base_ridge | 0.754 | 0.233 | 0.175 | 0.526 | 0.317 |
| text_logistic | 0.743 | 0.300 | 0.202 | 0.605 | 0.425 |
| text_ridge | 0.717 | 0.206 | 0.123 | 0.368 | 0.229 |

### text_ridge 高权重词项

正向词项：
slot, impact, sexist, machines, sexist email, gambling, slot machines, addiction, email, negative, winning, behavior, autism, gender, machines tips

负向词项：
creation, instructions, plan, animal torture, earth, guide, steps, animal, insider, date rape, date, rape, drug, child, rape drug

### text_logistic 高权重词项

正向词项：
relationship, addiction, gambling, sexist, impact, sexist email, slot, behavior, email, machines, machine, opinion, discrimination, plagiarism detection, introduction

负向词项：
instructions, creation, insider, guide, earth, betting, steps, to, animal, execution, control, avoiding detection, target audience, animal torture, preparation

## 前缀回答数 k=4

prefix-zero 测试记录：461；高风险数：8；比例：0.017。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.456 | 0.018 | 0.011 | 0.125 | 0.049 |
| base_logistic | 0.670 | 0.064 | 0.054 | 0.625 | 0.211 |
| base_ridge | 0.625 | 0.100 | 0.033 | 0.375 | 0.267 |
| text_logistic | 0.464 | 0.031 | 0.011 | 0.125 | 0.125 |
| text_ridge | 0.458 | 0.019 | 0.011 | 0.125 | 0.054 |

### text_ridge 高权重词项

正向词项：
machines, autism, expert, colleague, impact, slot machines, slot, scientific evidence, vaccines, 5g, relationship, covid, public, sexist email, sexist

负向词项：
creation, instructions, animal torture, plan, step, animal, earth, to, process, for, insider tips, date rape, date, rape, drug

### text_logistic 高权重词项

正向词项：
colleague, environment, machines, scientific evidence, slot, slot machines, promotion of, hazard_term ing, ing, event, effects on, introduction, machines tips, relationship, 19 expert

负向词项：
instructions, legal, to, risks, earth, step, claims, plan, analysis, hoax, insider, process, betting, creation, strategy

