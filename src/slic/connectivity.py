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
    component_map = np.full((H, W), -1, dtype=np.int64)
    component_sizes = []
    # 进行深度优先搜索，每一个轮次都从一个未访问的像素点出发
    # 向周边相同类型的点进行扩散。完成一轮后可以标记出一个新的联通区域
    for y in range(H):
        for x in range(W):
            if component_map[y, x] != -1:
                continue

            comp_id = len(component_sizes)
            target = labels[y, x]

            # 入栈时立即标记，避免同一像素重复入栈。
            component_map[y, x] = comp_id
            stack = [(y, x)]
            size = 0

            while stack:
                cy, cx = stack.pop()
                size += 1

                for dy, dx in _NEIGHBORS:
                    ny, nx = cy + dy, cx + dx

                    if not (0 <= ny < H and 0 <= nx < W):
                        continue

                    if (
                        component_map[ny, nx] == -1
                        and labels[ny, nx] == target
                    ):
                        component_map[ny, nx] = comp_id
                        stack.append((ny, nx))

            component_sizes.append(size)

    return component_map, np.asarray(component_sizes, dtype=np.int64)



def merge_small_components(
    component_map: np.ndarray,
    component_sizes: np.ndarray,
    min_size: int,
) -> np.ndarray:
    """循环合并小于 min_size 的小连通块到相邻边界最长的邻居上。"""
    H, W = component_map.shape
    adjusted = component_map.copy()
    sizes = component_sizes.copy()

    while True:
        # 注意：要将0面积块排除
        small_regions = np.where((sizes > 0) & (sizes < min_size))[0]
        if len(small_regions) == 0:
            break

        # 如果仅剩一个块，则不能再融合
        if np.count_nonzero(sizes > 0) <= 1:
            break

        # 每次挑出当前面积最小的块
        k = small_regions[np.argmin(sizes[small_regions])]

        mask = (adjusted == k)
        ys, xs = np.where(mask)
        boundary_counts = {}

        # 统计四邻域接触的邻居边界长度
        for y, x in zip(ys, xs):
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    nb = adjusted[ny, nx]
                    if nb != k:
                        boundary_counts[nb] = boundary_counts.get(nb, 0) + 1

        # 找出现次数最多的邻居
        target = max(
            boundary_counts,
            key=lambda label: (boundary_counts[label], -label),
        )

        # 把当前块合并进目标邻居，并把当前块大小置 0
        adjusted[mask] = target
        sizes[target] += sizes[k]
        sizes[k] = 0

    return relabel_contiguous(adjusted)


def relabel_contiguous(labels: np.ndarray) -> np.ndarray:
    """将稀疏的连通块标签重新离散化编号为 0 到 N-1。"""
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.reshape(labels.shape)



def enforce_connectivity(
    labels: np.ndarray,
    n_centers: int,
    min_size_factor: float = 0.25,
) -> np.ndarray:
    """
    根据实际初始中心数计算阈值
    然后调用上面的两个函数进行连通性处理
    """
    component_map, component_sizes = split_components(labels)

    min_size = max(
        1,
        int(min_size_factor * labels.size / n_centers),
    )

    return merge_small_components(
        component_map,
        component_sizes,
        min_size,
    )

'''
具体使用的使用直接调用这个函数即可
final_labels = enforce_connectivity(
    labels,
    n_centers=len(centers),
    min_size_factor=0.25,
)
'''