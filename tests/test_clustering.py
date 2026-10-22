"""像素距离竞争、局部分配、补分配和中心更新测试。"""

import numpy as np
import pytest

from slic.clustering import assign_pixels, update_centers


def test_color_distance_and_ties_choose_the_expected_center():
    lab = np.zeros((1, 3, 3), dtype=np.float64)
    lab[0, :, 0] = [0, 50, 100]
    centers = np.array([[0, 0, 0, 0, 0], [100, 0, 0, 0, 2]], dtype=np.float64)
    original_lab = lab.copy()
    original_centers = centers.copy()

    labels = assign_pixels(lab, centers, step=3, compactness=0)

    # 中间像素的颜色距离相等，保留较小编号 0。
    np.testing.assert_array_equal(labels, [[0, 0, 1]])
    assert np.issubdtype(labels.dtype, np.integer)
    np.testing.assert_array_equal(lab, original_lab)
    np.testing.assert_array_equal(centers, original_centers)


def test_same_color_pixels_follow_spatial_distance():
    lab = np.zeros((1, 5, 3), dtype=np.float64)
    centers = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 4]], dtype=np.float64)

    labels = assign_pixels(lab, centers, step=10, compactness=10)

    np.testing.assert_array_equal(labels, [[0, 0, 0, 1, 1]])


def test_assignment_preserves_fractional_center_coordinates():
    lab = np.zeros((1, 3, 3), dtype=np.float64)
    centers = np.array([[0, 0, 0, 0, 0.8], [0, 0, 0, 0, 1.6]], dtype=np.float64)

    labels = assign_pixels(lab, centers, step=3, compactness=1)

    # x=1 到两个真实中心的距离为 0.2 和 0.6；提前取整会改变归属。
    np.testing.assert_array_equal(labels, [[0, 0, 1]])


@pytest.mark.parametrize("compactness, expected", [(5, 0), (20, 1)])
def test_compactness_changes_color_spatial_tradeoff(compactness, expected):
    lab = np.zeros((1, 21, 3), dtype=np.float64)
    lab[0, 16, 0] = 2
    centers = np.array([[0, 0, 0, 0, 0], [8, 0, 0, 0, 12]], dtype=np.float64)

    labels = assign_pixels(lab, centers, step=20, compactness=compactness)

    # x=16：m=5 时距离平方为 20、37；m=20 时为 260、52。
    assert labels[0, 16] == expected


def test_uncovered_pixels_are_assigned_using_color_and_position():
    lab = np.zeros((1, 7, 3), dtype=np.float64)
    lab[0, 3:, 0] = 100
    centers = np.array([[0, 0, 0, 0, 0], [100, 0, 0, 0, 6]], dtype=np.float64)

    # 中间多个像素不在任一中心的局部窗口内，需要触发补分配。
    labels = assign_pixels(lab, centers, step=1, compactness=1)

    np.testing.assert_array_equal(labels, [[0, 0, 0, 1, 1, 1, 1]])
    assert np.all(labels >= 0)


def test_repeated_assignment_uses_the_new_centers():
    lab = np.zeros((1, 5, 3), dtype=np.float64)
    centers = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 4]], dtype=np.float64)

    first = assign_pixels(lab, centers, step=10, compactness=1)
    second = assign_pixels(lab, centers[::-1].copy(), step=10, compactness=1)

    np.testing.assert_array_equal(first, [[0, 0, 0, 1, 1]])
    np.testing.assert_array_equal(second, [[1, 1, 0, 0, 0]])


def test_update_centers_matches_hand_calculated_means():
    lab = np.array(
        [
            [[10, 1, 2], [20, 3, 4], [30, 5, 6]],
            [[40, 7, 8], [50, 9, 10], [60, 11, 12]],
        ],
        dtype=np.float64,
    )
    labels = np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int32)
    old_centers = np.zeros((2, 5), dtype=np.float64)
    original_lab = lab.copy()
    original_labels = labels.copy()
    original_centers = old_centers.copy()

    result = update_centers(lab, labels, old_centers)

    expected = [
        [70 / 3, 11 / 3, 14 / 3, 1 / 3, 1 / 3],
        [140 / 3, 25 / 3, 28 / 3, 2 / 3, 5 / 3],
    ]
    np.testing.assert_allclose(result, expected, atol=1e-12)
    np.testing.assert_array_equal(lab, original_lab)
    np.testing.assert_array_equal(labels, original_labels)
    np.testing.assert_array_equal(old_centers, original_centers)


def test_update_centers_preserves_empty_cluster():
    lab = np.full((2, 2, 3), [50, 5, -3], dtype=np.float64)
    labels = np.zeros((2, 2), dtype=np.int32)
    old_centers = np.array([[0, 0, 0, 0, 0], [80, 8, -6, 1, 1]], dtype=np.float64)

    result = update_centers(lab, labels, old_centers)

    np.testing.assert_allclose(result[0], [50, 5, -3, 0.5, 0.5])
    np.testing.assert_array_equal(result[1], old_centers[1])
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("step", [0, -1, np.nan, np.inf])
def test_assignment_rejects_invalid_step(step):
    with pytest.raises(ValueError):
        assign_pixels(np.zeros((1, 1, 3)), np.zeros((1, 5)), step, 10)


@pytest.mark.parametrize("compactness", [-1, np.nan, np.inf])
def test_assignment_rejects_invalid_compactness(compactness):
    with pytest.raises(ValueError):
        assign_pixels(np.zeros((1, 1, 3)), np.zeros((1, 5)), 1, compactness)


def test_assignment_rejects_an_empty_center_array():
    with pytest.raises(ValueError):
        assign_pixels(np.zeros((1, 1, 3)), np.empty((0, 5)), 1, 10)
