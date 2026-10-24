"""显著性结果的灰度图、前景查询二值图和对比展示"""

import numpy as np
import matplotlib.pyplot as plt

from .superpixels import _chinese_font


def saliency_image(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    将超像素显著性分数映射成像素级灰度图
    第一阶段和最终阶段都使用这个函数，保留分数原有的 [0, 1] 范围
    """
    return scores[labels]


def foreground_query_image(
    initial_scores: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    根据第一阶段分数生成前景查询二值图
    不低于超像素平均显著值的节点设为 True，与前景查询的选择规则一致
    显示时白色表示入选的前景查询区域，黑色表示其余区域
    """

    saliency_map = initial_scores[labels]
    threshold = saliency_map.mean()
    return saliency_map > threshold  # 顺手把 >= 写成了 >


def plot_saliency(image: np.ndarray, title: str = "显著图"):
    """
    展示单张显著图或前景查询二值图，返回 Figure
    """
    font = _chinese_font()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray")
    ax.set_title(title, fontproperties=font, fontsize=13)
    ax.axis("off")
    plt.tight_layout()

    return fig


def plot_stage_comparison(
    initial_saliency: np.ndarray,
    final_saliency: np.ndarray,
    title: str = None,
):
    """
    并排展示第一阶段与最终显著图，返回 Figure
    """
    font = _chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 真人只有两张图时，通常直接复制粘贴两遍，不会用 zip
    axes[0].imshow(initial_saliency, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("第一阶段显著图", fontproperties=font, fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(initial_saliency, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("最终显著图", fontproperties=font, fontsize=13)
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontproperties=font, fontsize=14)
    fig.tight_layout()

    return fig


def plot_saliency_comparison(
    original: np.ndarray,
    initial_saliency: np.ndarray,
    foreground_queries: np.ndarray,
    final_saliency: np.ndarray,
    title: str = None,
):
    """
    并排展示原图、第一阶段显著图、前景查询二值图和最终显著图
    接收分别生成好的三张结果图，返回 Figure 供调用方保存或显示
    """
    font = _chinese_font()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    imgs = [original, foreground_queries, initial_saliency, final_saliency]
    titles = ["原图", "第一阶段显著图", "前景查询二值图", "最终显著图"]

    for i in range(4):
        if i == 0:
            axes[i].imshow(imgs[i])
        else:
            axes[i].imshow(imgs[i], cmap="gray")
        axes[i].set_title(titles[i], fontproperties=font, fontsize=13)
        axes[i].axis("off")

    if title:
        plt.suptitle(title, fontproperties=font, fontsize=14)

    plt.tight_layout()
    return fig