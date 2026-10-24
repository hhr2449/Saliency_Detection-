"""超像素与显著性结果可视化的对外接口"""

from .superpixels import (
    find_boundaries,
    overlay_boundaries,
    mean_color_image,
    plot_comparison,
)
from .saliency import (
    saliency_image,
    foreground_query_image,
    plot_saliency,
    plot_stage_comparison,
    plot_saliency_comparison,
)

__all__ = [
    "find_boundaries",
    "overlay_boundaries",
    "mean_color_image",
    "plot_comparison",
    "saliency_image",
    "foreground_query_image",
    "plot_saliency",
    "plot_stage_comparison",
    "plot_saliency_comparison",
]
