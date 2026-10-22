"""SLIC 算法顶层接口。"""

import numpy as np

from .image import rgb_to_lab
from .initialization import compute_gradient, initialize_centers, perturb_centers
from .clustering import assign_pixels, update_centers
from .connectivity import split_components, merge_small_components


def slic(
    image: np.ndarray,
    n_segments: int = 200,
    compactness: float = 10.0,
    max_iter: int = 10,
    min_size_factor: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB 图像的 SLIC 超像素分割。"""
    lab = rgb_to_lab(image)

    # 增加参数检查
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(
        max_iter, (int, np.integer)
    ):
        raise TypeError("max_iter 必须是整数。")

    if max_iter < 1:
        raise ValueError("max_iter 必须大于等于 1。")
    if not np.isfinite(min_size_factor) or min_size_factor < 0:
        raise ValueError("min_size_factor 必须是有限的非负数。")

    # 1. 初始化聚类中心并微调
    grad = compute_gradient(lab)
    centers, step = initialize_centers(lab, n_segments)
    centers = perturb_centers(lab, centers, grad)
    k_init = len(centers)

    # 2. 局部 K-Means 迭代更新
    for _ in range(max_iter):
        labels = assign_pixels(lab, centers, step, compactness)
        centers = update_centers(lab, labels, centers)

    # 3. 连通域分离与小块合并
    comp_map, comp_sizes = split_components(labels)
    min_size = int(min_size_factor * (labels.size / k_init))
    
    final_labels = merge_small_components(comp_map, comp_sizes, max(1, min_size))

    # 4. 统计最终超像素的特征与大小
    final_sizes = np.bincount(final_labels.ravel())
    n_final = len(final_sizes)

    # 计算最终中心 (L, a, b, y, x)
    final_centers = np.zeros((n_final, 5), dtype=np.float64)
    ys, xs = np.indices(lab.shape[:2])
    
    for k in range(n_final):
        mask = (final_labels == k)
        if np.any(mask):
            final_centers[k, :3] = lab[mask].mean(axis=0)
            final_centers[k, 3] = ys[mask].mean()
            final_centers[k, 4] = xs[mask].mean()

    return final_centers, final_labels, final_sizes