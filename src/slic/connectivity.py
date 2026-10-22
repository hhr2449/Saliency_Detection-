"""SLIC 的连通性后处理。

提取独立连通块及面积，逐次合并当前最小的小区域，
最后将标签重新连续编号。
"""

from __future__ import annotations

import numpy as np


_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def split_components(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """提取四连通块，返回连通块编号图和各块面积。"""
    H, W = labels.shape
    component_map = np.full((H, W), -1, dtype=int)
    component_sizes = []

    # 进行深度优先搜索，每一个轮次都从一个未访问的像素点出发
    # 向周边相同类型的点进行扩散。完成一轮后可以标记出一个新的联通区域
    for y in range(H):
        for x in range(W):
            if component_map[y, x] != -1:
                continue

            comp_id = len(component_sizes)
            target = labels[y, x]

            stack = [(y, x)]
            size = 0
            while stack:
                cy, cx = stack.pop()
                if component_map[cy, cx] != -1:
                    continue
                component_map[cy, cx] = comp_id
                size += 1

                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if component_map[ny, nx] == -1 and labels[ny, nx] == target:
                            stack.append((ny, nx))

            component_sizes.append(size)

    return component_map, np.array(component_sizes)

