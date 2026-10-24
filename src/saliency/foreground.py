"""根据第一阶段显著分数选择前景查询，并计算最终节点显著性。"""

from __future__ import annotations

import numpy as np

from ranking import rank_queries

from .utils import normalize_scores


def foreground_query(
    ranking_matrix: np.ndarray,
    initial_scores: np.ndarray,
) -> np.ndarray:
    """
    选择不低于平均显著值的节点，进行前景查询并归一化
    """
    initial_scores = np.asarray(initial_scores, dtype=np.float64)
    if (
        initial_scores.ndim != 1
        or initial_scores.size == 0
        or not np.all(np.isfinite(initial_scores))
    ):
        raise ValueError("initial_scores 必须是非空的一维有限分数数组。")

    # 1. 用平均显著值选择前景候选，构造查询向量 y
    threshold = initial_scores.mean()
    query_vector = (initial_scores >= threshold).astype(np.float64)

    # 2. 复用关联矩阵，计算所有节点与前景候选的关联分数
    scores = rank_queries(ranking_matrix, query_vector)

    # 3. 前景关联越强越显著，直接归一化
    return normalize_scores(scores)
