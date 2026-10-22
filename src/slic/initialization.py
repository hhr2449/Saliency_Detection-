"""网格中心初始化及低梯度位置调整。

中心统一使用 [L, a, b, y, x] 顺序。
"""

from __future__ import annotations

import numpy as np

def compute_gradient(lab: np.ndarray) -> np.ndarray:
    """计算 Lab 图像的梯度平方强度。

    内部像素使用中心差分：
        dy = (I[y+1, x] - I[y-1, x]) / 2
        dx = (I[y, x+1] - I[y, x-1]) / 2
        G = ||dy||² + ||dx||²

    边界使用单边差分。
    某个空间维度长度为 1 时，该方向的导数取 0。

    返回 (H, W) 的 float64 数组。
    """
    image = np.asarray(lab, dtype=np.float64)
    H, W, _ = lab.shape

    diff_y = np.zeros_like(image)
    diff_x = np.zeros_like(image)
    
    if H > 1:
        # 内部：中心差分；边界：单边差分。
        diff_y[1:-1] = (image[2:] - image[:-2]) / 2.0
        diff_y[0] = image[1] - image[0]
        diff_y[-1] = image[-1] - image[-2]

    if W > 1:
        diff_x[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2.0
        diff_x[:, 0] = image[:, 1] - image[:, 0]
        diff_x[:, -1] = image[:, -1] - image[:, -2]
    return np.sum(diff_y**2 + diff_x**2, axis=-1)


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

    # 列数确定后，再根据目标数量调整行数。
    n_rows = int(np.clip(round(n_segments / n_cols), 1, H))

    # 取每个网格单元的中心位置
    # 这里需要根据实际行列数计算
    ys = ((np.arange(n_rows) + 0.5) * H / n_rows).astype(int)
    xs = ((np.arange(n_cols) + 0.5) * W / n_cols).astype(int)

    centers = []
    for y in ys:
        for x in xs:
            # 顺序: [L, a, b, y, x]
            center = np.zeros(5, dtype=np.float64)
            center[:3] = lab[y, x]
            center[3] = y
            center[4] = x
            centers.append(center)

    return np.asarray(centers, dtype=np.float64), step




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