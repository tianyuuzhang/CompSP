# Prefix-zero 隐蔽高风险二阶段模型

高风险阈值：后缀 ASR >= 0.5；选择比例：top 20%。
文本视图：`response`。

## 前缀回答数 k=1

prefix-zero 测试记录：903；高风险数：106；比例：0.117。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.475 | 0.111 | 0.105 | 0.179 | 0.217 |
| base_logistic | 0.735 | 0.230 | 0.271 | 0.462 | 0.451 |
| base_ridge | 0.742 | 0.243 | 0.304 | 0.519 | 0.457 |
| text_logistic | 0.768 | 0.232 | 0.238 | 0.406 | 0.431 |
| text_ridge | 0.774 | 0.245 | 0.276 | 0.472 | 0.458 |

### text_ridge 高权重词项

正向词项：
post, and, their, blog, blog post, pineapple, post that, network, while, social, individuals, article, of, an, write

负向词项：
refusal_term, end_header_id refusal_term, flat, refusal_term provide, planet, the planet, provide, provide evidence, evidence, request, sexist, pedophilia, sexist email, list, guidance on

### text_logistic 高权重词项

正向词项：
and, post, pineapple, blog, blog post, social, to, draft, design, group of, those, tax, while, to design, of people

负向词项：
flat, planet, the planet, refusal_term, mango, sexist, provide evidence, refusal_term provide, pedophilia, ransomware, evidence, sexist email, end_header_id refusal_term, list, provide

## 前缀回答数 k=2

prefix-zero 测试记录：776；高风险数：30；比例：0.039。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.462 | 0.039 | 0.045 | 0.233 | 0.080 |
| base_logistic | 0.769 | 0.104 | 0.148 | 0.767 | 0.249 |
| base_ridge | 0.800 | 0.125 | 0.161 | 0.833 | 0.289 |
| text_logistic | 0.810 | 0.110 | 0.129 | 0.667 | 0.242 |
| text_ridge | 0.808 | 0.112 | 0.155 | 0.800 | 0.267 |

### text_ridge 高权重词项

正向词项：
post, blog, blog post, post that, and, kiwi, article, gambling, their, financial, topic, write blog, write, an, explain

负向词项：
refusal_term, end_header_id refusal_term, refusal_term provide, the planet, planet, flat, provide, provide evidence, sexist, pedophilia, email, sexist email, evidence, request, assistant

### text_logistic 高权重词项

正向词项：
including gambling, gambling, individuals, with 回答分隔, and using, and, risky, scam, promote, refusal_term answer, post, attacks is, using, step explanation, write an

负向词项：
ransomware, provide, refusal_term provide, planet, the planet, or, pedophilia, sexist, refusal_term, there something, flat, refusal_term with, that 回答分隔, sexist email, someone

## 前缀回答数 k=4

prefix-zero 测试记录：698；高风险数：5；比例：0.007。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.516 | 0.010 | 0.007 | 0.200 | 0.026 |
| base_logistic | 0.760 | 0.056 | 0.029 | 0.800 | 0.138 |
| base_ridge | 0.949 | 0.095 | 0.036 | 1.000 | 0.200 |
| text_logistic | 0.851 | 0.027 | 0.029 | 0.800 | 0.069 |
| text_ridge | 0.919 | 0.049 | 0.036 | 1.000 | 0.106 |

### text_ridge 高权重词项

正向词项：
article, kiwi, is are, an article, blog, blog post, hazard_term, post, what is, what, write an, when, their, it, are kiwi

负向词项：
refusal_term, end_header_id refusal_term, refusal_term provide, provide, flat, pedophilia, sexist, evidence, or, sexist email, assistant, provide evidence, start_header_id assistant, start_header_id, refusal_term with

### text_logistic 高权重词项

正向词项：
refusal_term you, scam, individuals, including how, that 回答分隔, deceive, you with, individuals 回答分隔, fulfill that, fulfill, refusal_term fulfill, others, you, create explicit, contains

负向词项：
or, refusal_term with, refusal_term provide, ransomware, something, provide, guidance on, guidance, provide information, something else, on, on how, this request, or guidance, that promotes

