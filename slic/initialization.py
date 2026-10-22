"""网格中心初始化及低梯度位置调整。

中心统一使用 [L, a, b, y, x] 顺序。
"""

from __future__ import annotations

import numpy as np

def compute_gradient(lab: np.ndarray) -> np.ndarray:
    """
    计算 Lab 图像的梯度强度 
    使用水平、竖直方向颜色差的平方和：
    G = ||I[y+1, x] - I[y-1, x]||²
      + ||I[y, x+1] - I[y, x-1]||²

    返回 (H, W) 的 float64 数组。
    """
    H, W, _ = lab.shape
    grad = np.zeros((H, W), dtype=np.float64)

    diff_y = lab[2:, 1:-1] - lab[:-2, 1:-1]
    diff_x = lab[1:-1, 2:] - lab[1:-1, :-2]
    
    grad[1:-1, 1:-1] = np.sum(diff_y**2 + diff_x**2, axis=-1)
    return grad


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




def perturb_centers(
    lab: np.ndarray,
    centers: np.ndarray,
    gradient: np.ndarray,
) -> np.ndarray:
    """
    将中心移动到 3x3 邻域内梯度最小的位置。
    """
    H, W = lab.shape[:2]
    adjusted = centers.copy()

    for i in range(len(adjusted)):
        y = int(adjusted[i, 3])
        x = int(adjusted[i, 4])

        min_grad = gradient[y, x]
        best_y, best_x = y, x

        y_min = max(0, y - 1)
        y_max = min(H, y + 2)
        x_min = max(0, x - 1)
        x_max = min(W, x + 2)

        # 找 3x3 窗口内最小梯度的坐标
        local_grad = gradient[y_min:y_max, x_min:x_max]
        min_idx = np.unravel_index(np.argmin(local_grad), local_grad.shape)

        cand_y = y_min + min_idx[0]
        cand_x = x_min + min_idx[1]

        # 如果找到更小梯度的位置就更新
        if gradient[cand_y, cand_x] < min_grad:
            adjusted[i, 3] = cand_y
            adjusted[i, 4] = cand_x
            adjusted[i, :3] = lab[cand_y, cand_x]

    return adjusted