"""非归一化流形排序的关联矩阵与查询分数计算。"""

from __future__ import annotations

import numpy as np


def compute_ranking_matrix(
    weights: np.ndarray,
    alpha: float = 0.99,
) -> np.ndarray:
    """
    计算 B = (D - alpha * W)^(-1)，再将 B 的对角线清零

    weights 是建图输出的 (N, N) 对称非负权重矩阵
    alpha 为衰减系数，满足 0 < alpha < 1

    因为后面的背景查询和前景查询都要利用相同的B，为了实现复用提供获取B的接口
    """

    # 1. 参数检查：W为方阵，alpha的取值范围，过滤孤立节点
    W = np.asarray(weights, dtype=np.float64)
    assert W.ndim == 2 and W.shape[0] == W.shape[1], "W 必须是方阵"
    assert 0 < alpha < 1, "alpha 需在 (0, 1) 之间"

    degrees = W.sum(axis=1)
    active = degrees > 0  # 过滤度为0的孤立节点，防止矩阵不可逆

    B = np.zeros_like(W)
    if not np.any(active):
        return B

    # 2. 在非零度子图上求解 (D - alpha * W)^(-1)
    sub_W = W[np.ix_(active, active)]
    sub_M = np.diag(degrees[active]) - alpha * sub_W
    B[np.ix_(active, active)] = np.linalg.inv(sub_M)

    # 3. 抹除节点自身对自身的关联
    np.fill_diagonal(B, 0.0)
    return B



