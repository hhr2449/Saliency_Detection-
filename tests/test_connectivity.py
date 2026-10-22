"""四连通块拆分、面积统计、动态小块合并和连续编号测试。"""

import numpy as np
import pytest

from slic.connectivity import (
    enforce_connectivity,
    merge_small_components,
    relabel_contiguous,
    split_components,
)


def test_split_components_separates_disconnected_parts(assert_four_connected):
    labels = np.array([[5, 5, 9, 5], [5, 9, 9, 5], [7, 7, 9, 5]])
    original = labels.copy()

    component_map, sizes = split_components(labels)

    assert component_map.shape == labels.shape
    assert component_map[0, 0] == component_map[1, 0]
    assert component_map[0, 0] != component_map[0, 3]
    assert component_map[0, 3] == component_map[2, 3]
    np.testing.assert_array_equal(np.sort(sizes), [2, 3, 3, 4])
    np.testing.assert_array_equal(sizes, np.bincount(component_map.ravel()))
    assert sizes.sum() == labels.size
    assert np.all(component_map >= 0)
    assert_four_connected(component_map)
    for k in range(len(sizes)):
        assert np.unique(labels[component_map == k]).size == 1
    np.testing.assert_array_equal(labels, original)


def test_diagonal_contact_is_not_four_connected(assert_four_connected):
    component_map, sizes = split_components(np.array([[0, 1], [1, 0]]))

    assert np.unique(component_map).size == 4
    np.testing.assert_array_equal(sizes, [1, 1, 1, 1])
    assert_four_connected(component_map)


def test_area_count_does_not_count_repeated_visits():
    component_map, sizes = split_components(np.full((3, 4), 42))

    np.testing.assert_array_equal(component_map, np.zeros((3, 4), dtype=int))
    np.testing.assert_array_equal(sizes, [12])


@pytest.mark.parametrize("shape", [(1, 4), (4, 1)])
def test_split_components_handles_thin_images(shape, assert_four_connected):
    labels = np.array([0, 0, 1, 0]).reshape(shape)

    component_map, sizes = split_components(labels)

    np.testing.assert_array_equal(np.sort(sizes), [1, 1, 2])
    assert_four_connected(component_map)


def test_small_island_merges_without_mutating_inputs(assert_four_connected):
    component_map = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    sizes = np.array([8, 1])
    original_map = component_map.copy()
    original_sizes = sizes.copy()

    result = merge_small_components(component_map, sizes, min_size=2)

    np.testing.assert_array_equal(result, np.zeros((3, 3), dtype=int))
    np.testing.assert_array_equal(component_map, original_map)
    np.testing.assert_array_equal(sizes, original_sizes)
    assert_four_connected(result)


def test_small_region_chooses_the_longest_shared_boundary(assert_four_connected):
    component_map = np.array([[0, 1, 1], [0, 2, 1], [0, 1, 1]])

    result = merge_small_components(component_map, np.array([3, 5, 1]), min_size=2)

    # 区域 2 与区域 1 接触三条边，与区域 0 只接触一条边。
    assert result[1, 1] == result[0, 1]
    assert result[1, 1] != result[1, 0]
    assert_four_connected(result)


def test_equal_boundary_lengths_choose_the_smaller_region_id():
    """回归：并列时按编号选择，而不是按遍历时首次遇到的邻区选择。"""
    component_map = np.array(
        [[0, 0, 0, 0], [1, 1, 2, 2], [1, 1, 3, 2], [1, 1, 1, 2]]
    )

    result = merge_small_components(component_map, np.array([4, 7, 4, 1]), min_size=2)

    # 区域 3 与区域 1、2 各接触两条边，应选择区域 1。
    assert result[2, 2] == result[2, 1]
    assert result[2, 2] != result[1, 2]


def test_merge_reselects_the_current_smallest_region(assert_four_connected):
    """A 并入 B 后 B 变大，应先处理 C，不能沿用初始面积排序。"""
    component_map = np.array([[0] * 6 + [1] * 3 + [2] * 2 + [3] * 4 + [4] * 6])
    sizes = np.array([6, 3, 2, 4, 6])

    result = merge_small_components(component_map, sizes, min_size=6)

    # 先 2→1，面积变为 5；再 3→1，面积变为 9，留下三个区域。
    expected = np.array([[0] * 6 + [1] * 9 + [2] * 6])
    np.testing.assert_array_equal(result, expected)
    assert_four_connected(result)


def test_equal_small_areas_are_processed_in_region_id_order():
    component_map = np.array([[0] * 4 + [1] + [2] + [3] * 4])

    result = merge_small_components(component_map, np.array([4, 1, 1, 4]), min_size=2)

    # 先处理 1 后处理 2；若先处理 2，会留下一个不同的中间区域。
    np.testing.assert_array_equal(result, [[0] * 6 + [1] * 4])


def test_region_equal_to_threshold_is_not_merged():
    component_map = np.array([[0, 0, 1, 1, 1]])

    result = merge_small_components(component_map, np.array([2, 3]), min_size=2)

    np.testing.assert_array_equal(result, component_map)


def test_single_remaining_region_stops_below_threshold():
    component_map = np.zeros((2, 2), dtype=int)

    result = merge_small_components(component_map, np.array([4]), min_size=10)

    np.testing.assert_array_equal(result, component_map)


def test_all_small_components_can_merge_into_one_region(assert_four_connected):
    component_map, sizes = split_components(np.array([[0, 1, 0], [1, 0, 1]]))

    result = merge_small_components(component_map, sizes, min_size=10)

    np.testing.assert_array_equal(result, np.zeros((2, 3), dtype=int))
    assert_four_connected(result)


def test_relabel_contiguous_preserves_the_partition():
    labels = np.array([[8, 8, 20], [3, 20, 3]])
    original = labels.copy()

    result = relabel_contiguous(labels)

    np.testing.assert_array_equal(result, [[1, 1, 2], [0, 2, 0]])
    np.testing.assert_array_equal(np.unique(result), [0, 1, 2])
    np.testing.assert_array_equal(labels, original)


@pytest.mark.parametrize("factor, expected_count", [(0, 2), (0.5, 2), (0.74, 2), (0.75, 1)])
def test_enforce_connectivity_uses_reference_area_and_floor(factor, expected_count):
    labels = np.array([[0] * 6 + [1] * 2])

    result = enforce_connectivity(labels, n_centers=2, min_size_factor=factor)

    assert np.unique(result).size == expected_count
    np.testing.assert_array_equal(np.unique(result), np.arange(expected_count))
