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
    """在各中心约 2S x 2S 的局部邻域内更新像素标签。"""
    H, W = lab.shape[:2]
    K = len(centers)

    
    labels = np.zeros((H, W), dtype=np.int32)
    distances = np.full((H, W), 1e10, dtype=np.float64)

    S = int(step)
    inv_step2 = (compactness / step) ** 2

    for k in range(K):
        cy = int(centers[k, 3])
        cx = int(centers[k, 4])

        y_min = max(0, cy - S)
        y_max = min(H, cy + S)
        x_min = max(0, cx - S)
        x_max = min(W, cx + S)

        patch_lab = lab[y_min:y_max, x_min:x_max]
        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]

        # 算颜色距离和空间距离
        d_color = np.sum((patch_lab - centers[k, :3]) ** 2, axis=-1)
      
        d_spatial = (yy - cy) ** 2 + (xx - cx) ** 2
        d_total = d_color + inv_step2 * d_spatial

        sub_dist = distances[y_min:y_max, x_min:x_max]
        mask = d_total < sub_dist
        sub_dist[mask] = d_total[mask]
        labels[y_min:y_max, x_min:x_max][mask] = k

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