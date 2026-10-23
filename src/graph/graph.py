"""
根据 SLIC 最终标签和平均 Lab 颜色构建超像素图。

1. 输入标签图，计算区域邻接关系
2. 在区域邻接关系的基础上加上邻居的邻居的连接
3. 加上边缘区域的全相联
4. 计算边权
5. 全流程串联，最终对外接口就是输入labels矩阵和平均颜色，输出邻接矩阵和边权矩阵
"""

from __future__ import annotations

import numpy as np


def build_spatial_adjacency(labels: np.ndarray) -> np.ndarray:
    """提取相邻超像素的空间邻接矩阵"""
    H, W = labels.shape
    n_nodes = int(labels.max()) + 1
    adj = np.zeros((n_nodes, n_nodes), dtype=bool)

    for y in range(H):
        for x in range(W):
            u = labels[y, x]

            # 检查下方邻居
            if y + 1 < H:
                v = labels[y + 1, x]
                if u != v:
                    adj[u, v] = True
                    adj[v, u] = True

            # 检查右侧邻居
            if x + 1 < W:
                v = labels[y, x + 1]
                if u != v:
                    adj[u, v] = True
                    adj[v, u] = True

    return adj


def add_two_hop_connections(spatial_adj: np.ndarray) -> np.ndarray:
    """添加二跳连接 (A + A^2)"""
    # 计算 A^2
    two_hop = (spatial_adj.astype(float) @ spatial_adj.astype(float)) > 0
    # 和 A 取或
    adj = spatial_adj | two_hop
    np.fill_diagonal(adj, False)
    return adj


def add_boundary_connections(
    labels: np.ndarray,
    adj: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """将位于图像四周边缘的超像素节点全部两两互连"""

    # 先提取四个边上的区域编号
    boundary_nodes = {
        "top": np.unique(labels[0, :]),
        "bottom": np.unique(labels[-1, :]),
        "left": np.unique(labels[:, 0]),
        "right": np.unique(labels[:, -1]),
    }
    # 然后去重
    all_boundary = np.unique(np.concatenate(list(boundary_nodes.values())))

    
    # 边缘节点全连接
    for u in all_boundary:
        for v in all_boundary:
            if u != v:
                adj[u, v] = True

    return adj, boundary_nodes


def compute_weights(
    adj: np.ndarray,
    mean_lab: np.ndarray,
    sigma_square: float,
) -> np.ndarray:
    """
    计算已连接节点间的颜色高斯相似度权重
    """
    weights = np.zeros(adj.shape, dtype=np.float64)
    rows, cols = np.nonzero(adj)

    # 无边图直接返回零矩阵，避免对空距离数组求最值。
    if rows.size == 0:
        return weights

    # 1. 仅计算存在边的节点对距离
    dist = np.linalg.norm(mean_lab[rows] - mean_lab[cols], axis=1)
    # 2. 新增：将距离归一化到 [0, 1]。
    d_min = dist.min()
    d_max = dist.max()
    if d_max > d_min:
        dist = (dist - d_min) / (d_max - d_min)
    else:
        # 所有距离相等时，采用统一边权为 1 的约定。
        dist = np.zeros_like(dist)

    # 3. 使用归一化后的距离计算权重。
    weights[rows, cols] = np.exp(-dist / sigma_square)

    return weights


def build_graph(
    labels: np.ndarray,
    mean_lab: np.ndarray,
    sigma_square: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    构建完整的超像素图
    """
    adj = build_spatial_adjacency(labels)
    adj = add_two_hop_connections(adj)
    adj, boundary_nodes = add_boundary_connections(labels, adj)
    weights = compute_weights(adj, mean_lab, sigma_square)

    return adj, weights, boundary_nodes
