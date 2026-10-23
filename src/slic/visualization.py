
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

