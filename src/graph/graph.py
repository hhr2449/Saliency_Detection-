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


