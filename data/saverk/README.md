# saverk

CompSP 的 pairwise 排序输出。

每个 `save_*` 文件夹都存放按问题划分的 pairwise 矩阵和 Bradley-Terry 排序分数。典型的问题子文件夹会包含：

- `Q_<id>_<start>_<end>pair.json`：pairwise 矩阵，其中单元格 `(i,j)` 估计 `prompt i` 排在 `prompt j` 前的概率。
- `Q_<id>_<start>_<end>_ranking_scores.json`：Bradley-Terry 分数和按分数排序后的 prompt ID。
