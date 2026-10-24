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


def rank_queries(
    ranking_matrix: np.ndarray,
    query_vector: np.ndarray,
) -> np.ndarray:
    """
    根据关联矩阵 B 和查询向量 y 计算排序分数 f = B @ y

    ranking_matrix 是已清零对角线的 (N, N) 关联矩阵
    query_vector 的形状为 (N,)，查询节点位置为 1，其他位置为 0

    返回 (N,) 的 float64 原始关联分数，归一化由查询阶段处理
    输入矩阵和查询向量保持不变
    """
    B = np.asarray(ranking_matrix, dtype=np.float64)
    y = np.asarray(query_vector, dtype=np.float64)

    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError(f"ranking_matrix 需为二维方阵，当前 shape: {B.shape}")
    if y.shape != (B.shape[0],):
        raise ValueError(f"query_vector 长度 {y.shape} 与节点数 {B.shape[0]} 不一致")

    return B @ y
