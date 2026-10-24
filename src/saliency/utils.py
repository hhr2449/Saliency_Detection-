"""显著性计算共用的分数归一化。"""

from __future__ import annotations

import numpy as np


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """将非空分数数组归一化到 [0, 1]，相同分数约定返回全零。"""
    scores = np.asarray(scores, dtype=np.float64)
    score_min = scores.min()
    score_max = scores.max()

    if score_max == score_min:
        return np.zeros_like(scores)

    return (scores - score_min) / (score_max - score_min)
