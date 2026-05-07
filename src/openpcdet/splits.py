import random
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class DatasetSplits:
    train: List[dict]
    val: List[dict]
    test: List[dict]


def train_val_test_split(infos: Sequence[dict], val_ratio: float, test_ratio: float, seed: int) -> DatasetSplits:
    if val_ratio < 0.0 or test_ratio < 0.0:
        raise ValueError("Split ratios must be >= 0")
    if val_ratio + test_ratio >= 1.0 and len(infos) > 1:
        raise ValueError("val_ratio + test_ratio must be < 1")

    indices = list(range(len(infos)))
    random.Random(seed).shuffle(indices)

    num_test = int(round(len(indices) * test_ratio))
    num_val = int(round(len(indices) * val_ratio))

    if len(indices) >= 3:
        num_test = max(1, num_test) if test_ratio > 0.0 else 0
        num_val = max(1, num_val) if val_ratio > 0.0 else 0
        while num_test + num_val >= len(indices):
            if num_val >= num_test and num_val > 0:
                num_val -= 1
            elif num_test > 0:
                num_test -= 1
            else:
                break
    else:
        num_test = min(num_test, max(0, len(indices) - 1))
        num_val = min(num_val, max(0, len(indices) - num_test - 1))

    test_indices = set(indices[:num_test])
    val_indices = set(indices[num_test : num_test + num_val])

    train = []
    val = []
    test = []

    for index, info in enumerate(infos):
        if index in test_indices:
            test.append(info)
        elif index in val_indices:
            val.append(info)
        else:
            train.append(info)

    return DatasetSplits(train=train, val=val, test=test)
