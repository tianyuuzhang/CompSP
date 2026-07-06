import json
from collections import defaultdict

# 读取原始数据
with open('/remote-home/zty/save/merged_questions.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 构建 qid -> list of tx 映射
qid_to_entries = defaultdict(list)
for item in data:
    qid_to_entries[item["questionID"]].append(item)

# 数据集1：所有 questionID
all_qids_dataset = [entry for entries in qid_to_entries.values() for entry in entries]

# 数据集2：仅包含至少一个 tx >= 5 的 questionID
qids_with_tx_ge_5 = {
    qid for qid, entries in qid_to_entries.items() if any(e["tx"] >= 5 for e in entries)
}
qids_ge_5_dataset = [entry for qid, entries in qid_to_entries.items() if qid in qids_with_tx_ge_5 for entry in entries]

# 保存两个数据集
with open('/remote-home/zty/save/all_qids_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(all_qids_dataset, f, ensure_ascii=False, indent=2)

with open('/remote-home/zty/save/qids_with_tx_ge_5_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(qids_ge_5_dataset, f, ensure_ascii=False, indent=2)

# 统计 tx 分布函数
def compute_tx_distribution(dataset):
    tx_count = defaultdict(int)
    for item in dataset:
        tx_count[item["tx"]] += 1
    return dict(sorted(tx_count.items()))

# 输出分布
print("🔎 所有 QID 数据集 tx 分布：", compute_tx_distribution(all_qids_dataset))
print("🔎 tx≥5 QID 数据集 tx 分布：", compute_tx_distribution(qids_ge_5_dataset))