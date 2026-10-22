"""SLIC 的距离计算、局部像素分配和聚类中心更新。

中心统一使用 [L, a, b, y, x] 顺序。
像素分配输出原始中心编号；连通性整理和最终连续编号由后续模块完成。
"""

from __future__ import annotations

import numpy as np


def assign_pixels(
    lab: np.ndarray,
    centers: np.ndarray,
    step: float,
    compactness: float,
) -> np.ndarray:
    """在各中心约 2S × 2S 的局部邻域内分配像素。

    返回：
        (H, W) 的 int32 标签图，标签是 centers 的行号。

    未被窗口覆盖的像素补算到全部中心的距离。
    距离相同时，在参与比较的中心中保留编号较小者。
    """
    H, W = lab.shape[:2]
    K = len(centers)

    if K == 0:
        raise ValueError("至少需要一个聚类中心。")

    if not np.isfinite(step) or step <= 0:
        raise ValueError("step 必须是有限的正数。")

    if not np.isfinite(compactness) or compactness < 0:
        raise ValueError("compactness 必须是有限的非负数。")

    labels = np.full((H, W), -1, dtype=np.int32)
    distances = np.full((H, W), np.inf, dtype=np.float64)

    spatial_weight = (compactness / step) ** 2

    for k in range(K):
        # 保留真实的浮点中心坐标。
        cy = centers[k, 3]
        cx = centers[k, 4]

        # 仅在确定数组切片范围时取整。
        y_min = max(0, int(np.floor(cy - step)))
        y_max = min(H, int(np.ceil(cy + step)))
        x_min = max(0, int(np.floor(cx - step)))
        x_max = min(W, int(np.ceil(cx + step)))

        patch_lab = lab[y_min:y_max, x_min:x_max]
        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]

        d_color = np.sum(
            (patch_lab - centers[k, :3]) ** 2,
            axis=-1,
        )
        d_spatial = (yy - cy) ** 2 + (xx - cx) ** 2
        d_total = d_color + spatial_weight * d_spatial

        sub_dist = distances[y_min:y_max, x_min:x_max]
        sub_labels = labels[y_min:y_max, x_min:x_max]

        mask = d_total < sub_dist
        sub_dist[mask] = d_total[mask]
        sub_labels[mask] = k

    # 补充分配未被任何局部窗口覆盖的像素。
    missing_y, missing_x = np.where(labels < 0)

    for y, x in zip(missing_y, missing_x):
        d_color = np.sum(
            (centers[:, :3] - lab[y, x]) ** 2,
            axis=1,
        )
        d_spatial = (
            (centers[:, 3] - y) ** 2
            + (centers[:, 4] - x) ** 2
        )
        d_total = d_color + spatial_weight * d_spatial
        labels[y, x] = np.argmin(d_total)

    return labels


def update_centers(
    lab: np.ndarray,
    labels: np.ndarray,
    old_centers: np.ndarray,
) -> np.ndarray:
    """根据像素聚类结果重新计算簇均值作为新中心。"""
    K = len(old_centers)
    new_centers = np.zeros_like(old_centers)

    for k in range(K):
        mask = (labels == k)
        
        if np.any(mask):
            new_centers[k, :3] = lab[mask].mean(axis=0)
            
            ys, xs = np.where(mask)
            new_centers[k, 3] = ys.mean()
            new_centers[k, 4] = xs.mean()
        else:
            new_centers[k] = old_centers[k]

    return new_centers