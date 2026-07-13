# Prefix-zero 隐蔽高风险二阶段模型

高风险阈值：后缀 ASR >= 0.5；选择比例：top 20%。
文本视图：`joint`。

## 前缀回答数 k=1

prefix-zero 测试记录：903；高风险数：106；比例：0.117。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.475 | 0.111 | 0.105 | 0.179 | 0.217 |
| base_logistic | 0.735 | 0.230 | 0.271 | 0.462 | 0.451 |
| base_ridge | 0.742 | 0.243 | 0.304 | 0.519 | 0.457 |
| text_logistic | 0.688 | 0.219 | 0.188 | 0.321 | 0.297 |
| text_ridge | 0.723 | 0.229 | 0.210 | 0.358 | 0.355 |

### text_ridge 高权重词项

正向词项：
post, article, blog, blog post, network, their, individuals, kiwi, end_header_id user, of, post that, write blog, an article, social media, while

负向词项：
refusal_term, end_header_id refusal_term, with that, request, that request, refusal_term with, flat, ransomware, pedophilia, provide evidence, question, the planet, planet, draft step, per

### text_logistic 高权重词项

正向词项：
post, article, while, blog, draft instructions, social, social media, blog post, media, network, to outline, outline, preparing banana, the thing, avoiding

负向词项：
request, refusal_term, end_header_id refusal_term, that request, with that, refusal_term with, flat, planet, the planet, pedophilia, ransomware, provide evidence, draft step, evidence, per

## 前缀回答数 k=2

prefix-zero 测试记录：776；高风险数：30；比例：0.039。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.462 | 0.039 | 0.045 | 0.233 | 0.080 |
| base_logistic | 0.769 | 0.104 | 0.148 | 0.767 | 0.249 |
| base_ridge | 0.800 | 0.125 | 0.161 | 0.833 | 0.289 |
| text_logistic | 0.769 | 0.088 | 0.097 | 0.500 | 0.193 |
| text_ridge | 0.805 | 0.111 | 0.155 | 0.800 | 0.273 |

### text_ridge 高权重词项

正向词项：
post, blog, blog post, article, end_header_id user, post that, an article, write blog, kiwi, post question, when, essential to, individuals, it essential, essential

负向词项：
refusal_term, end_header_id refusal_term, with that, refusal_term with, request, that request, request 回答分隔, question, banana, pedophilia, flat, the planet, planet, refusal_term provide, ransomware

### text_logistic 高权重词项

正向词项：
gambling, individuals, including gambling, post, preparing banana, apples, article, framework, step explanation, an official, item referred, approach outlined, set, mangos, thing in

负向词项：
refusal_term, with that, end_header_id refusal_term, refusal_term with, cherry, planet, the planet, ransomware, that request, flat, request, pedophilia, banana, phrased, vivid narrative

## 前缀回答数 k=4

prefix-zero 测试记录：698；高风险数：5；比例：0.007。

| 策略 | ROC-AUC | AP | top precision | top recall | max F1 |
|---|---:|---:|---:|---:|---:|
| random | 0.516 | 0.010 | 0.007 | 0.200 | 0.026 |
| base_logistic | 0.760 | 0.056 | 0.029 | 0.800 | 0.138 |
| base_ridge | 0.949 | 0.095 | 0.036 | 1.000 | 0.200 |
| text_logistic | 0.936 | 0.071 | 0.036 | 1.000 | 0.174 |
| text_ridge | 0.939 | 0.074 | 0.036 | 1.000 | 0.174 |

### text_ridge 高权重词项

正向词项：
article, an article, end_header_id user, kiwi, blog, blog post, post, when, are kiwi, article question, write blog, impact, write an, article about, next

负向词项：
end_header_id refusal_term, refusal_term, request, with that, refusal_term with, that request, request 回答分隔, question, with, assistant, start_header_id, start_header_id assistant, assistant end_header_id, end_header_id, refusal_term provide

### text_logistic 高权重词项

正向词项：
individuals, article, item referred, networks, an article, central theme, to in, blog post, blog, write blog, from their, people into, referred to, referred, including how

负向词项：
item referenced, end_header_id refusal_term, referenced, referenced in, refusal_term with, refusal_term, with that, here is, sample, is guide, order, request, request 回答分隔, that request, sure here

