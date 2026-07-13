# Prefix-zero 隐蔽高风险二阶段模型

高风险阈值：后缀 ASR >= 0.5；选择比例：top 20%。
文本视图：`q1`。

## 前缀回答数 k=1

prefix-zero 测试记录：903；高风险数：106；比例：0.117。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.475 | 0.111 | 0.105 | 0.179 | 0.217 |
| base_logistic | 0.735 | 0.230 | 0.271 | 0.462 | 0.451 |
| base_ridge | 0.742 | 0.243 | 0.304 | 0.519 | 0.457 |
| text_logistic | 0.437 | 0.116 | 0.083 | 0.142 | 0.215 |
| text_ridge | 0.418 | 0.106 | 0.077 | 0.132 | 0.213 |

### text_ridge 高权重词项

正向词项：
post, blog, article, post question, kiwi, blog post, network, business network, business, write blog, step how, article question, if, why, watermelon

负向词项：
ransomware, pedophilia, email question, person, deploying, email, target, grape, deploying ransomware, draft step, linguistic substitution, step, per, access, memo

### text_logistic 高权重词项

正向词项：
post, blog, network, article, business network, business, blog post, while, post question, draft instructions, preparing banana, step how, thing, bread, the thing

负向词项：
ransomware, pedophilia, email question, target, step, person, explains how, draft step, email, grape, deploying, deploying ransomware, evidence, substitution, sexist email

## 前缀回答数 k=2

prefix-zero 测试记录：776；高风险数：30；比例：0.039。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.462 | 0.039 | 0.045 | 0.233 | 0.080 |
| base_logistic | 0.769 | 0.104 | 0.148 | 0.767 | 0.249 |
| base_ridge | 0.800 | 0.125 | 0.161 | 0.833 | 0.289 |
| text_logistic | 0.486 | 0.052 | 0.032 | 0.167 | 0.093 |
| text_ridge | 0.444 | 0.042 | 0.019 | 0.100 | 0.087 |

### text_ridge 高权重词项

正向词项：
post, article, kiwi, blog, post question, blog post, article question, write blog, an article, why, policy, conduct, tracking, mango, tracking conduct

负向词项：
pedophilia, step, instructions question, ransomware, apple, instructions, step instructions, person, email question, email, by, script, draft step, banana, memo

### text_logistic 高权重词项

正向词项：
article, network, preparing banana, business network, business, post, gambling, describe, blog, playful, post question, to describe, kiwi, librarian, mangos

负向词项：
question for, ransomware, cherry, pedophilia, vivid, to apply, phrased, person, apply the, craft vivid, phrased as, specified, deploying, the expected, by

## 前缀回答数 k=4

prefix-zero 测试记录：698；高风险数：5；比例：0.007。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.516 | 0.010 | 0.007 | 0.200 | 0.026 |
| base_logistic | 0.760 | 0.056 | 0.029 | 0.800 | 0.138 |
| base_ridge | 0.949 | 0.095 | 0.036 | 1.000 | 0.200 |
| text_logistic | 0.414 | 0.008 | 0.007 | 0.200 | 0.025 |
| text_ridge | 0.455 | 0.008 | 0.000 | 0.000 | 0.019 |

### text_ridge 高权重词项

正向词项：
kiwi, article, an article, article question, are kiwi, brief, policy brief, post, mango, subject of, kiwi question, user query, policy, general, blog

负向词项：
clear, instructions question, and, instructions, pedophilia, apple, email, item referenced, step instructions, describing how, hidden, hidden meanings, pineapple, draft step, item

### text_logistic 高权重词项

正向词项：
kiwi, article, referred, referred to, are kiwi, policy brief, brief, an article, answer centered, vocabulary, article question, item referred, write blog, of documents, the thing

负向词项：
item referenced, apply the, to apply, instructions, the resulting, resulting, banana, sample, referenced, by, referenced in, apply, here is, resulting answer, is guide

