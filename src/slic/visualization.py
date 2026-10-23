
from __future__ import annotations

import numpy as np

from .image import prepare_rgb
import matplotlib.pyplot as plt

'''
SLIC 结果可视化
主要展示两种图像：
1. 在原图上标出边界
2. 将区域替换为平均颜色
'''


def find_boundaries(labels: np.ndarray) -> np.ndarray:
    """
    提取区域边界
    输入标签图labels，输出boundaries，其为boolean数组，对应每个像素是否是边界
    分别进行行列的比较，对比相邻的行、列，只要相应位置的标签不同，就说明是边界
    """
    boundaries = np.zeros(labels.shape, dtype=bool)

    vertical = labels[1:, :] != labels[:-1, :]
    boundaries[1:, :] |= vertical
    boundaries[:-1, :] |= vertical

    horizontal = labels[:, 1:] != labels[:, :-1]
    boundaries[:, 1:] |= horizontal
    boundaries[:, :-1] |= horizontal

    return boundaries


def overlay_boundaries(
    rgb: np.ndarray,
    labels: np.ndarray,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> np.ndarray:
    """
    在原图副本上叠加边界，返回 [0, 1] 的 RGB float32 图像。
    直接将边界的像素改为指定的颜色
    """
    result = prepare_rgb(rgb)

    result[find_boundaries(labels)] = color
    return result


def mean_color_image(rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """将各区域替换为其原始 RGB 像素的平均颜色。"""
    img = prepare_rgb(rgb)
    out = np.zeros_like(img)

    for k in range(labels.max() + 1):
        mask = labels == k
        out[mask] = img[mask].mean(axis=0)

    return out

