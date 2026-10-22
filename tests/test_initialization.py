"""纯色、线性变化、颜色边缘及低梯度中心调整测试。"""

import numpy as np
import pytest

from slic.initialization import compute_gradient, initialize_centers, perturb_centers


@pytest.mark.parametrize("shape", [(6, 8), (1, 7), (7, 1), (1, 1)])
def test_uniform_image_has_zero_gradient(shape):
    lab = np.full((*shape, 3), [50, 5, -3], dtype=np.float64)

    gradient = compute_gradient(lab)

    assert gradient.shape == shape
    np.testing.assert_array_equal(gradient, np.zeros(shape))


def test_uniform_image_centers_form_an_even_grid():
    lab = np.full((8, 12, 3), [50, 5, -3], dtype=np.float64)
    original = lab.copy()

    centers, step = initialize_centers(lab, n_segments=6)

    assert step == pytest.approx(4)
    assert centers.shape == (6, 5)
    expected_positions = [[2, 2], [2, 6], [2, 10], [6, 2], [6, 6], [6, 10]]
    np.testing.assert_array_equal(centers[:, 3:], expected_positions)
    np.testing.assert_array_equal(centers[:, :3], np.tile([50, 5, -3], (6, 1)))
    np.testing.assert_array_equal(lab, original)


@pytest.mark.parametrize(
    "H, W, expected",
    [(5, 7, 35), (2, 3, 35), (1, 7, 29), (7, 1, 6), (1, 1, 0)],
)
def test_linear_image_gradient_matches_hand_calculation(H, W, expected):
    yy, xx = np.indices((H, W))
    lab = np.stack((2 * yy + 3 * xx, -yy + 4 * xx, yy - 2 * xx), axis=-1)
    lab = lab.astype(np.float64)

    # y 方向平方和为 2²+(-1)²+1²=6，x 方向为 3²+4²+(-2)²=29。
    gradient = compute_gradient(lab)

    np.testing.assert_allclose(gradient, np.full((H, W), expected), atol=1e-12)


def test_gradient_detects_a_sharp_color_edge():
    lab = np.zeros((5, 6, 3), dtype=np.float64)
    lab[:, 3:, 0] = 100
    expected = np.zeros((5, 6))
    expected[:, 2:4] = 2500

    np.testing.assert_allclose(compute_gradient(lab), expected)


@pytest.mark.parametrize(
    "start, target",
    [((2, 2), (1, 3)), ((0, 0), (0, 1)), ((4, 4), (3, 4))],
)
def test_perturb_centers_moves_to_lower_gradient_and_updates_color(start, target):
    yy, xx = np.indices((5, 5))
    lab = np.stack((10 * yy + xx, yy, xx), axis=-1).astype(np.float64)
    y, x = start
    centers = np.array([[*lab[y, x], y, x]], dtype=np.float64)
    original_centers = centers.copy()
    original_lab = lab.copy()
    gradient = np.full((5, 5), 10.0)
    gradient[target] = 0
    original_gradient = gradient.copy()

    adjusted = perturb_centers(lab, centers, gradient)

    np.testing.assert_array_equal(adjusted[0, 3:], target)
    np.testing.assert_array_equal(adjusted[0, :3], lab[target])
    np.testing.assert_array_equal(centers, original_centers)
    np.testing.assert_array_equal(lab, original_lab)
    np.testing.assert_array_equal(gradient, original_gradient)


def test_perturb_centers_stays_in_place_when_gradients_are_equal():
    lab = np.zeros((8, 12, 3), dtype=np.float64)
    centers, _ = initialize_centers(lab, 6)

    adjusted = perturb_centers(lab, centers, compute_gradient(lab))

    np.testing.assert_array_equal(adjusted, centers)


def test_perturb_centers_only_searches_the_local_neighborhood():
    lab = np.zeros((5, 5, 3), dtype=np.float64)
    centers = np.array([[0, 0, 0, 2, 2]], dtype=np.float64)
    gradient = np.full((5, 5), 10.0)
    gradient[0, 0] = 0  # 全图最低点，但不在中心的 3×3 邻域内。
    gradient[2, 3] = 1

    adjusted = perturb_centers(lab, centers, gradient)

    np.testing.assert_array_equal(adjusted[0, 3:], [2, 3])


@pytest.mark.parametrize("shape", [(1, 1), (5, 5), (1, 25), (25, 1)])
def test_one_requested_segment_initializes_one_center(shape):
    """回归：长条图的网格取整也应保留 K=1 的语义。"""
    lab = np.zeros((*shape, 3), dtype=np.float64)

    centers, step = initialize_centers(lab, 1)

    assert centers.shape == (1, 5)
    assert step == pytest.approx(np.sqrt(shape[0] * shape[1]))
    assert 0 <= centers[0, 3] < shape[0]
    assert 0 <= centers[0, 4] < shape[1]


def test_one_center_per_pixel_has_unique_valid_positions():
    lab = np.arange(36, dtype=np.float64).reshape(3, 4, 3)

    centers, step = initialize_centers(lab, 12)

    assert step == pytest.approx(1)
    assert len(np.unique(centers[:, 3:], axis=0)) == 12
    for center in centers:
        y, x = center[3:].astype(int)
        assert 0 <= y < 3 and 0 <= x < 4
        np.testing.assert_array_equal(center[:3], lab[y, x])


@pytest.mark.parametrize("n_segments", [0, -1, 13])
def test_initialize_centers_rejects_out_of_range_segment_counts(n_segments):
    with pytest.raises(ValueError):
        initialize_centers(np.zeros((3, 4, 3), dtype=np.float64), n_segments)
