# Prefix-zero 隐蔽高风险二阶段模型

高风险阈值：后缀 ASR >= 0.5；选择比例：top 20%。
文本视图：`q1`。

## 前缀回答数 k=1

prefix-zero 测试记录：619；高风险数：108；比例：0.174。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.518 | 0.170 | 0.145 | 0.167 | 0.321 |
| base_logistic | 0.773 | 0.438 | 0.379 | 0.435 | 0.513 |
| base_ridge | 0.763 | 0.431 | 0.379 | 0.435 | 0.501 |
| text_logistic | 0.730 | 0.349 | 0.331 | 0.380 | 0.441 |
| text_ridge | 0.470 | 0.201 | 0.113 | 0.130 | 0.313 |

### text_ridge 高权重词项

正向词项：
post, story, character, relationships, blog, blog post, article, write, creative, these, personal, young, unconventional, historical, might

负向词项：
earth, sports, flat, evidence, football, the earth, tips, arguments, on how, insider, how to, theory, is flat, certain, shape

### text_logistic 高权重词项

正向词项：
post, character, relationships, unconventional, impact, blog, blog post, find, personal, might, historian, young, story, readers, in an

负向词项：
earth, sports, flat, tips, the earth, football, evidence, theories, how to, theory, strategies, arguments, group, thread, consultant

## 前缀回答数 k=2

prefix-zero 测试记录：501；高风险数：33；比例：0.066。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.512 | 0.067 | 0.050 | 0.152 | 0.146 |
| base_logistic | 0.609 | 0.110 | 0.150 | 0.455 | 0.257 |
| base_ridge | 0.480 | 0.068 | 0.080 | 0.242 | 0.142 |
| text_logistic | 0.440 | 0.089 | 0.070 | 0.212 | 0.200 |
| text_ridge | 0.337 | 0.047 | 0.000 | 0.000 | 0.130 |

### text_ridge 高权重词项

正向词项：
story, refusal_term, refusal_term refusal_term, article, q1_generation_failed_a0_error refusal_term, q1_generation_failed_a0_error, post, the story, character, these, with that, refusal_term with, events, methods, actions

负向词项：
earth, sports, flat, football, the earth, evidence, financial, consultant, to, insider, information, provide, campaign, arguments, is flat

### text_logistic 高权重词项

正向词项：
character, story, post, article, unconventional, relationships, setting, childhood, the story, these, people, alternate, thoughts, one, product

负向词项：
provide, they, earth, sports, flat, can, consultant, you provide, football, are an, of this, be, the earth, campaign, certain

## 前缀回答数 k=4

prefix-zero 测试记录：409；高风险数：3；比例：0.007。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.615 | 0.017 | 0.012 | 0.333 | 0.047 |
| base_logistic | 0.566 | 0.020 | 0.012 | 0.333 | 0.071 |
| base_ridge | 0.370 | 0.008 | 0.000 | 0.000 | 0.021 |
| text_logistic | 0.406 | 0.008 | 0.000 | 0.000 | 0.024 |
| text_ridge | 0.487 | 0.009 | 0.000 | 0.000 | 0.026 |

### text_ridge 高权重词项

正向词项：
refusal_term, refusal_term refusal_term, q1_generation_failed_a0_error, q1_generation_failed_a0_error refusal_term, with that, refusal_term with, events, captivating, that, story, article, show, write compelling, medieval, art

负向词项：
earth, financial, campaign, football, flat, sports, evidence, the earth, would, their, without, improvement, certain, consultant, for

### text_logistic 高权重词项

正向词项：
refusal_term, medieval, into, refusal_term refusal_term, journalist, art, blog, article, era, the era, you please, ability, captivating, re participating, are journalist

负向词项：
their, creative, would, but, world, for, public, earth, present, without, need to, part, to create, part of, must

