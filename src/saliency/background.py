"""分别查询图像四侧背景，融合得到第一阶段显著性分数。"""

from __future__ import annotations

import numpy as np

from ranking import rank_queries

from .utils import normalize_scores


def background_query(
    ranking_matrix: np.ndarray,
    boundary_nodes: dict[str, np.ndarray],
) -> np.ndarray:
    """
    四侧分别排序、归一化取反，再相乘并归一化

    ranking_matrix 是矩阵 B，可使用ranking模块中的接口获取并复用
    boundary_nodes 包含 top、bottom、left、right 四组节点编号，在之前建图的时候就获取了这个字典，可以复用
    
    """
    B = np.asarray(ranking_matrix)

    n_nodes = B.shape[0]
    initial_scores = np.ones(n_nodes, dtype=np.float64)

    # 每一侧分别查询，并将显著性分数逐元素相乘
    for side in ("top", "bottom", "left", "right"):
        query_vector = np.zeros(n_nodes, dtype=np.float64)
        query_vector[boundary_nodes[side]] = 1.0

        scores = rank_queries(B, query_vector)
        side_saliency = 1.0 - scores
        initial_scores *= side_saliency

    # 对融合结果归一化，供前景阶段选择查询节点
    scores = np.asarray(scores, dtype=np.float64)
    score_min = scores.min()
    score_max = scores.max()

    if score_max == score_min:
        return np.zeros_like(scores)

    return (scores - score_min) / (score_max - score_min)
