"""ASR/ALR 实验数据集注册表。

源文件 `/remote-home/zty/tidy_CompSP/data/asr数据` 是人工可读索引。
本模块是代码使用的机器可读版本。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ASRDatasetSpec:
    key: str
    dataset: str
    attack: str
    target_model: str
    path_template: str
    expected_records_per_file: int = 100
    expected_tot_qa: int = 20
    notes: str = ""

    def path_for(self, question_id: int) -> Path:
        return Path(self.path_template.format(i=question_id))

    @property
    def root_dir(self) -> Path:
        marker = "/Q_{i}/"
        if marker in self.path_template:
            return Path(self.path_template.split(marker, 1)[0])
        return Path(self.path_template).parent


ASR_DATASETS: dict[str, ASRDatasetSpec] = {
    "jbb-4omini-ofa": ASRDatasetSpec(
        key="jbb-4omini-ofa",
        dataset="jbb",
        attack="ofa",
        target_model="4omini",
        path_template="/remote-home/xzh/for_zty/gpt3dot5_data_test/4omini/save/Q_{i}/Q_r_{i}_0_100_txfilled.json",
    ),
    "jbb-deepseek-ofa": ASRDatasetSpec(
        key="jbb-deepseek-ofa",
        dataset="jbb",
        attack="ofa",
        target_model="deepseek",
        path_template="/remote-home/xzh/for_zty/gpt3dot5_data_test/deepseek/save/Q_{i}/Q_{i}_0_100_txfilled.json",
        notes="JBB deepseek OFA txfilled 文件不使用 Q_r_ 前缀。",
    ),
    "jbb-llama-ofa": ASRDatasetSpec(
        key="jbb-llama-ofa",
        dataset="jbb",
        attack="ofa",
        target_model="llama",
        path_template="/remote-home/xzh/for_zty/gpt3dot5_data_test/llama/save/Q_{i}/Q_r_{i}_0_100_txfilled.json",
    ),
    "jbb-qwen-ofa": ASRDatasetSpec(
        key="jbb-qwen-ofa",
        dataset="jbb",
        attack="ofa",
        target_model="qwen",
        path_template="/remote-home/xzh/for_zty/gpt3dot5_data_test/qwen/save/Q_{i}/Q_r_{i}_0_100_txfilled.json",
    ),
    "hb-4omini-ofa": ASRDatasetSpec(
        key="hb-4omini-ofa",
        dataset="hb",
        attack="ofa",
        target_model="4omini",
        path_template="/remote-home/xzh/for_zty/gpt3dot5_400data_test/4omini/save/Q_{i}/Q_r_{i}_0_100_txfilled.json",
    ),
    "hb-deepseek-ofa": ASRDatasetSpec(
        key="hb-deepseek-ofa",
        dataset="hb",
        attack="ofa",
        target_model="deepseek",
        path_template="/remote-home/xzh/for_zty/gpt3dot5_400data_test/deepseek/save/Q_{i}/Q_r_{i}_0_100_txfilled.json",
    ),
    "hb-llama-ofa": ASRDatasetSpec(
        key="hb-llama-ofa",
        dataset="hb",
        attack="ofa",
        target_model="llama",
        path_template="/remote-home/xzh/for_zty/gpt3dot5_400data_test/llama/save/Q_{i}/Q_r_{i}_0_100_txfilled.json",
    ),
    "hb-qwen-ofa": ASRDatasetSpec(
        key="hb-qwen-ofa",
        dataset="hb",
        attack="ofa",
        target_model="qwen",
        path_template="/remote-home/xzh/for_zty/gpt3dot5_400data_test/qwen/save/Q_{i}/Q_r_{i}_0_100_txfilled.json",
    ),
    "jbb-qwen-drattack": ASRDatasetSpec(
        key="jbb-qwen-drattack",
        dataset="jbb",
        attack="drattack",
        target_model="qwen",
        path_template="/remote-home/xzh/for_zty/reproduce/drattack/DrAttack/Instructions_rewrite/save/Q_{i}/Q_r_{i}_0_100_txfilled.json",
        notes="当前目录中预期只存在 Q_0..Q_79。",
    ),
    "jbb-4omini-drattack": ASRDatasetSpec(
        key="jbb-4omini-drattack",
        dataset="jbb",
        attack="drattack",
        target_model="4omini",
        path_template="/remote-home/xzh/for_zty/reproduce/drattack/DrAttack/Instructions_rewrite_4omini/save/Q_{i}/Q_r_{i}_0_100_txfilled.json",
        notes="当前目录中预期只存在 Q_0..Q_79。",
    ),
    "jbb-llama-drattack": ASRDatasetSpec(
        key="jbb-llama-drattack",
        dataset="jbb",
        attack="drattack",
        target_model="llama",
        path_template="/remote-home/xzh/for_zty/reproduce/drattack/DrAttack/Instructions_rewrite_llama/save/Q_{i}/Q_r_{i}_0_100_txfilled.json",
        notes="当前目录中预期只存在 Q_0..Q_79。",
    ),
    "jbb-qwen-pair": ASRDatasetSpec(
        key="jbb-qwen-pair",
        dataset="jbb",
        attack="pair",
        target_model="qwen",
        path_template="/remote-home/xzh/for_zty/reproduce/tasks/pair/save/Q_{i}/Q_r_{i}_0_100_txfilled.json",
        notes="当前目录中预期只存在 Q_0..Q_79。",
    ),
    "jbb-llama-pair": ASRDatasetSpec(
        key="jbb-llama-pair",
        dataset="jbb",
        attack="pair",
        target_model="llama",
        path_template="/remote-home/xzh/for_zty/reproduce/tasks/pair_llama/save/Q_{i}/Q_r_{i}_0_100_txfilled.json",
        notes="修正人工索引中的重复后缀。",
    ),
    "jbb-4omini-pair": ASRDatasetSpec(
        key="jbb-4omini-pair",
        dataset="jbb",
        attack="pair",
        target_model="4omini",
        path_template="/remote-home/xzh/for_zty/reproduce/tasks/pair_4omini/save/Q_{i}/Q_r_{i}_0_100_txfilled.json",
        notes="修正人工索引中的重复后缀。",
    ),
}


def get_asr_spec(key: str) -> ASRDatasetSpec:
    try:
        return ASR_DATASETS[key]
    except KeyError as exc:
        known = ", ".join(sorted(ASR_DATASETS))
        raise KeyError(f"未知 ASR 数据集 key {key!r}; 已知 key: {known}") from exc
