"""串联各算法模块，提供完整的显著性检测入口"""

from __future__ import annotations

import numpy as np

from graph import build_graph
from ranking import compute_ranking_matrix
from saliency import background_query, foreground_query
from slic import slic


def detect_saliency(
    image: np.ndarray,
    n_segments: int = 200,
    compactness: float = 10.0,
    max_iter: int = 10,
    min_size_factor: float = 0.25,
    sigma_square: float = 0.1,
    alpha: float = 0.99,
    *,
    return_details: bool = False,
) -> np.ndarray | dict[str, np.ndarray]:
    """
    串联 SLIC、超像素建图和两阶段查询，返回像素级显著图

    image 为 (H, W, 3) 的 RGB 图像
    sigma_square 为边权公式中的 sigma²，alpha 为排序参数

    默认返回 (H, W) 的 float64 显著图，取值范围为 [0, 1]
    return_details=True 时返回显著图、中心、标签、面积和两阶段节点分数
    输入图像保持不变
    """
    if not np.isfinite(sigma_square) or sigma_square <= 0:
        raise ValueError("sigma_square 必须是有限的正数。")

    # 1. SLIC 分割，获取最终标签和区域平均 Lab 颜色
    centers, labels, sizes = slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        max_iter=max_iter,
        min_size_factor=min_size_factor,
    )

    # 2. 构建超像素图，提取边权和四侧边界节点
    _, weights, boundary_nodes = build_graph(
        labels,
        centers[:, :3],
        sigma_square=sigma_square,
    )

    # 3. 只计算一次关联矩阵，供背景和前景查询复用
    B = compute_ranking_matrix(weights, alpha=alpha)

    # 4. 背景查询生成初步显著分数，再进行前景查询
    initial_scores = background_query(B, boundary_nodes)
    final_scores = foreground_query(B, initial_scores)

    # 5. 将超像素分数映射回每个像素
    saliency_map = final_scores[labels]

    if return_details:
        return {
            "saliency_map": saliency_map,
            "centers": centers,
            "labels": labels,
            "sizes": sizes,
            "initial_scores": initial_scores,
            "final_scores": final_scores,
        }

    return saliency_map
