"""可视化图像、中文标题及三联图保存测试。"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from slic.visualization import (
    find_boundaries,
    mean_color_image,
    overlay_boundaries,
    plot_comparison,
)


def test_visualization_images_and_three_panel_plot(tmp_path):
    """使用手算样例检查边界、RGB 均值、中文标题、并排布局和保存。"""
    image = np.array(
        [[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]] * 2,
        dtype=np.uint8,
    )
    labels = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
    original_image = image.copy()
    original_labels = labels.copy()

    boundary_mask = find_boundaries(labels)
    np.testing.assert_array_equal(
        boundary_mask,
        [[False, True, True, False], [False, True, True, False]],
    )

    boundaries = overlay_boundaries(image, labels)
    expected_boundaries = image.astype(np.float32) / 255
    expected_boundaries[:, 1:3] = [1, 0, 0]
    np.testing.assert_allclose(boundaries, expected_boundaries)

    mean_colors = mean_color_image(image, labels)
    expected_means = np.array(
        [[[0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, 0.5, 1], [0.5, 0.5, 1]]] * 2,
        dtype=np.float32,
    )
    np.testing.assert_allclose(mean_colors, expected_means)
    np.testing.assert_array_equal(image, original_image)
    np.testing.assert_array_equal(labels, original_labels)

    fig = plot_comparison(image, boundaries, mean_colors, title="测试图像")
    try:
        assert len(fig.axes) == 3
        assert [ax.get_title() for ax in fig.axes] == ["原图", "区域边界图", "区域平均颜色图"]
        x_positions = [ax.get_position().x0 for ax in fig.axes]
        assert x_positions[0] < x_positions[1] < x_positions[2]

        for ax, expected in zip(fig.axes, (image, boundaries, mean_colors)):
            np.testing.assert_allclose(ax.images[0].get_array(), expected)

        output_path = tmp_path / "comparison.png"
        fig.savefig(output_path)
        with Image.open(output_path) as saved:
            assert saved.width > saved.height
            saved.verify()
    finally:
        plt.close(fig)
