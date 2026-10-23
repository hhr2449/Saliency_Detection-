
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import font_manager
import matplotlib.pyplot as plt

from .image import prepare_rgb

'''
SLIC 结果可视化
主要展示三种图像：
1. 原始 RGB 图像
2. 在原图上标出边界
3. 将区域替换为平均颜色
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


def _chinese_font():
    """
    选择系统中文字体，也兼容 Matplotlib 字体缓存尚未更新的情况。
    """
    font_paths = font_manager.findSystemFonts()
    candidates = ("notosanscjk-regular", "sourcehansans", "msyh", "simhei", "pingfang", "droidsansfallback")

    for candidate in candidates:
        for font_path in font_paths:
            if candidate in Path(font_path).name.lower():
                return font_manager.FontProperties(fname=font_path)

    return font_manager.FontProperties(family=["Microsoft YaHei", "SimHei", "sans-serif"])


def plot_comparison(
    original: np.ndarray,
    boundaries: np.ndarray,
    mean_colors: np.ndarray,
    title: str | None = None,
):
    """
    将三幅图并排展示，返回 Figure，供调用方保存或显示。
    """

    font = _chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    images = (original, boundaries, mean_colors)
    titles = ("原图", "区域边界图", "区域平均颜色图")

    for ax, image, panel_title in zip(axes, images, titles):
        ax.imshow(image, interpolation="nearest")
        ax.set_title(panel_title, fontproperties=font, fontsize=13)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontproperties=font, fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.tight_layout()

    return fig
