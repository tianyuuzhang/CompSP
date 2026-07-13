# Prefix-zero 强掩码 response 的 SVD 语义探针

分量数：32；最大词项：12000；清理口径：`mask_strong_artifacts`。

| k | records | positive_rate | EVR sum | svd_ridge AP | svd_ridge recall | best1 AP | best1 recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2230 | 0.141 | 0.416 | 0.436 | 0.559 | 0.378 | 0.485 |
| 2 | 1853 | 0.054 | 0.450 | 0.187 | 0.590 | 0.231 | 0.517 |
| 4 | 1590 | 0.013 | 0.473 | 0.053 | 0.605 | 0.083 | 0.446 |

## 首个 seed 的代表分量词项

### k=1
- C0 EVR=0.072；正向：artifact_term, artifact_term artifact_term, with that, artifact_term with, that artifact_term, with, that, artifact_term _id；负向：passphrase, bombing, ordeal, homeless, valtoria, luminari, umbra collective, the account
- C1 EVR=0.081；正向：with that, artifact_term with, that artifact_term, that, artifact_term, fulfill that, fulfill, artifact_term fulfill；负向：artifact_term information, or guidance, guidance on, information or, else can, artifact_term activities, is there, on artifact_term
- C2 EVR=0.023；正向：the, to, and, of, this, in, it, for；负向：artifact_term activities, activities including, artifact_term with, or guidance, with that, on artifact_term, guidance on, information or
- C3 EVR=0.032；正向：fulfill that, artifact_term fulfill, fulfill, that artifact_term, artifact_term, artifact_term artifact_term, that, or guidance；负向：with that, artifact_term with, with, you with, that is, you, else, is there
- C4 EVR=0.024；正向：you with, else, else can, is there, can help, help you, there, help；负向：or guidance, on artifact_term, information or, guidance on, artifact_term activities, activities including, guidance, artifact_term or
- C5 EVR=0.021；正向：error, encountered, artifact_term more, error message, message, seems, seems like, it seems；负向：of, in, and, by, as, of the, for, are

### k=2
- C0 EVR=0.077；正向：artifact_term, artifact_term artifact_term, that artifact_term, with that, artifact_term with, artifact_term _id, _id, start_ artifact_term；负向：the rebels, the empire, secretive elite, alexander, arin, contestants, toroidal, pizza
- C1 EVR=0.078；正向：that artifact_term, with that, artifact_term with, that, artifact_term, fulfill that, fulfill, artifact_term fulfill；负向：else, else can, is there, artifact_term information, can help, artifact_term activities, activities including, help you
- C2 EVR=0.029；正向：the, to, and, of, this, it, in, as；负向：artifact_term activities, activities including, or guidance, on artifact_term, information or, guidance on, activities, artifact_term or
- C3 EVR=0.026；正向：on artifact_term, artifact_term activities, activities including, or guidance, information or, or artifact_term, artifact_term or, activities；负向：you with, else, is there, else can, can help, there, help you, you
- C4 EVR=0.025；正向：error, encountered, artifact_term more, error message, seems, seems like, it seems, message；负向：of, by, in, of the, as, are, for, evidence
- C5 EVR=0.023；正向：fulfill, artifact_term fulfill, fulfill that, else, anything else, there anything, anything, can help；负向：with that, artifact_term with, with, or guidance, information or, guidance on, activities including, artifact_term activities

### k=4
- C0 EVR=0.078；正向：artifact_term, artifact_term artifact_term, that artifact_term, with that, start_, start_ artifact_term, artifact_term _id, _id artifact_term；负向：thomas, enlightened, the argentum, the starlight, dr kim, cakes from, whisk in, temperature large
- C1 EVR=0.064；正向：that artifact_term, with that, artifact_term with, that, fulfill that, artifact_term, artifact_term fulfill, fulfill；负向：else, else can, is there, can help, with artifact_term, help you, there, help
- C2 EVR=0.032；正向：the, and, to, of, in, this, as, it；负向：activities including, artifact_term activities, on artifact_term, or guidance, information or, or artifact_term, activities, guidance on
- C3 EVR=0.026；正向：activities including, on artifact_term, artifact_term activities, or artifact_term, activities, artifact_term or, or guidance, information or；负向：information on, else, you with, on how, is there, else can, date rape, rape
- C4 EVR=0.022；正向：suicide, mental health, mental, health, 800, lifeline, health professional, suicide prevention；负向：rape, date rape, rape drugs, drugs, drugs is, date, something else, something
- C5 EVR=0.020；正向：earth, the earth, evidence, that the, shape, earth is, artifact_term earth, curvature；负向：your, betting, game, error, stay, team, bets, losses
