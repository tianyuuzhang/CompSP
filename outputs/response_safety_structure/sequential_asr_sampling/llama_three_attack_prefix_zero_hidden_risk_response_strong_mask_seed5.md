# Prefix-zero 隐蔽高风险二阶段模型

高风险阈值：后缀 ASR >= 0.5；选择比例：top 20%。
文本视图：`response`。
回答归属打乱：`none`。

## 前缀回答数 k=1

prefix-zero 测试记录：2219；高风险数：319；比例：0.144。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.508 | 0.147 | 0.146 | 0.204 | 0.254 |
| base_logistic | 0.750 | 0.299 | 0.306 | 0.426 | 0.451 |
| base_ridge | 0.749 | 0.297 | 0.304 | 0.423 | 0.453 |
| text_logistic | 0.789 | 0.419 | 0.374 | 0.520 | 0.460 |
| text_ridge | 0.792 | 0.425 | 0.401 | 0.558 | 0.470 |

### text_ridge 高权重词项

正向词项：
and, social, consequences, social media, who, step, individuals, media, aged, their, of, your artifact_term, addiction, business, network

负向词项：
artifact_term, artifact_term artifact_term, evidence, planet, artifact_term evidence, the planet, team, earth, shape, the earth, illuminati, the illuminati, game, that promotes, global events

### text_logistic 高权重词项

正向词项：
and, network, money, social, the artifact_term, your artifact_term, aged, to, database, so, life, design, viewing, tax, of

负向词项：
general, illuminati, the illuminati, planet, team, evidence, global events, artifact_term evidence, shape, different, the planet, date rape, rape, specific, that artifact_term

## 前缀回答数 k=2

prefix-zero 测试记录：1846；高风险数：101；比例：0.055。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.510 | 0.060 | 0.054 | 0.198 | 0.108 |
| base_logistic | 0.735 | 0.122 | 0.136 | 0.495 | 0.228 |
| base_ridge | 0.730 | 0.113 | 0.119 | 0.436 | 0.224 |
| text_logistic | 0.780 | 0.182 | 0.165 | 0.604 | 0.276 |
| text_ridge | 0.789 | 0.204 | 0.165 | 0.604 | 0.298 |

### text_ridge 高权重词项

正向词项：
consequences, social, addiction, it, artifact_term question, using, individuals, by, erotic, consequences of, your artifact_term, movie, gambling, reputation, long

负向词项：
artifact_term, artifact_term artifact_term, is artifact_term, planet, evidence, the planet, earth, that promotes, artifact_term evidence, email, football, pedophilia, on how, team, date

### text_logistic 高权重词项

正向词项：
gambling, materials, fear, cryptocurrency, cryptocurrency mining, network, mining, and using, with artifact_term, so, and, individuals, your artifact_term, eliminating, information is

负向词项：
artifact_term is, ransomware, on how, including hacking, including animal, artifact_term with, planet, pornography is, animal torture, the planet, cannibalism, animal, with that, creation of, email

## 前缀回答数 k=4

prefix-zero 测试记录：1568；高风险数：16；比例：0.010。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.396 | 0.008 | 0.000 | 0.000 | 0.021 |
| base_logistic | 0.692 | 0.031 | 0.025 | 0.500 | 0.091 |
| base_ridge | 0.738 | 0.044 | 0.029 | 0.562 | 0.105 |
| text_logistic | 0.781 | 0.111 | 0.035 | 0.688 | 0.174 |
| text_ridge | 0.843 | 0.144 | 0.041 | 0.812 | 0.207 |

### text_ridge 高权重词项

正向词项：
code, error, issue, it, artifact_term question, manipulation, context, an artifact_term, hacking is, encountered, is are, error code, using, more, include

负向词项：
that artifact_term, is artifact_term, artifact_term, that promotes, artifact_term with, artifact_term artifact_term, with that, that, evidence, promotes, football, teams, on how, including the, pedophilia artifact_term

### text_logistic 高权重词项

正向词项：
information is, enslaving, tax, artifact_term you, individuals, manipulation, atm, form of, form, deceive, including how, including suicide, artifact_term question, you with, code

负向词项：
artifact_term with, that, ransomware, including the, on how, that is, that promotes, promotes, with that, content that, with something, is artifact_term, earth, ransomware is, planet

