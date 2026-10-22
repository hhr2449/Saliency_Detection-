"""网格中心初始化及低梯度位置调整。

中心统一使用 [L, a, b, y, x] 顺序。
"""

from __future__ import annotations

import numpy as np




def initialize_centers(
    lab: np.ndarray,
    n_segments: int,
) -> tuple[np.ndarray, float]:
    
    """在规则网格中初始化中心。
    参数：
        lab：形状为 (H, W, 3) 的浮点 Lab 图像。
        n_segments：期望超像素数量 K，满足 1 <= K <= H*W。

    返回：
        centers：(K0, 5) 的 float64 数组，顺序为 [L, a, b, y, x]。
        step：参考间隔 S = sqrt(H*W/K)。
    """
    H, W = lab.shape[:2]
    if not (1 <= n_segments <= H * W):
        raise ValueError(f"n_segments ({n_segments}) 超出有效范围 [1, {H * W}]")

    step = float(np.sqrt(H * W / n_segments))

    # 计算行列数并截断在合法尺寸内
    n_rows = int(np.clip(round(H / step), 1, H))
    n_cols = int(np.clip(round(n_segments / n_rows), 1, W))

    # 取每个网格单元的中心位置
    ys = np.linspace(step / 2, H - step / 2, n_rows).astype(int)
    xs = np.linspace(step / 2, W - step / 2, n_cols).astype(int)

    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    colors = lab[grid_y, grid_x].reshape(-1, 3)

    centers = []
    for y in ys:
        for x in xs:
            # 顺序: [L, a, b, y, x]
            center = np.zeros(5, dtype=np.float64)
            center[:3] = lab[y, x]
            center[3] = y
            center[4] = x
            centers.append(center)

    return centers, step




