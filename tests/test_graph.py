"""超像素图的空间邻接、二阶连接、边界连接、边权及 SLIC 接口测试。

按当前实现约定，边权使用已连接节点之间的颜色欧氏距离，
先做 min-max 归一化，再计算 exp(-distance / sigma_square)。
距离全部相等时，已连接节点之间的权重约定为 1。
"""

import numpy as np
import pytest

from graph import build_graph
from graph.graph import (
    add_boundary_connections,
    add_two_hop_connections,
    build_spatial_adjacency,
    compute_weights,
)
from slic.core import slic


# 1. 直接空间邻接

def test_spatial_adjacency_on_a_rectangular_label_map():
    """只共享一个角的区域不能连接；矩形标签图也能正确处理。"""
    labels = np.array([[0, 0, 1], [2, 2, 3]])
    original = labels.copy()
    expected = np.array(
        [[False, True, True, False],
         [True, False, False, True],
         [True, False, False, True],
         [False, True, True, False]]
    )

    adjacency = build_spatial_adjacency(labels)

    assert adjacency.dtype == np.bool_
    np.testing.assert_array_equal(adjacency, expected)
    np.testing.assert_array_equal(labels, original)


def test_repeated_boundary_contacts_are_recorded_as_one_connection():
    labels = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]])

    adjacency = build_spatial_adjacency(labels)

    assert adjacency.dtype == np.bool_
    np.testing.assert_array_equal(adjacency, [[False, True], [True, False]])


@pytest.mark.parametrize("shape", [(1, 5), (5, 1)])
def test_spatial_adjacency_on_thin_images(shape):
    labels = np.arange(5).reshape(shape)
    expected = np.array(
        [[False, True, False, False, False],
         [True, False, True, False, False],
         [False, True, False, True, False],
         [False, False, True, False, True],
         [False, False, False, True, False]]
    )

    np.testing.assert_array_equal(build_spatial_adjacency(labels), expected)


@pytest.mark.parametrize("shape", [(1, 1), (3, 5)])
def test_single_region_has_no_spatial_self_connection(shape):
    labels = np.zeros(shape, dtype=np.int64)

    np.testing.assert_array_equal(build_spatial_adjacency(labels), [[False]])


# 2. 二阶邻接

def test_two_hop_connections_stop_at_two_steps():
    """A—B—C—D—E 只增加相隔两步的连接，不继续递归扩展。"""
    direct = np.zeros((5, 5), dtype=bool)
    for k in range(4):
        direct[k, k + 1] = direct[k + 1, k] = True
    original = direct.copy()
    expected = np.array(
        [[False, True, True, False, False],
         [True, False, True, True, False],
         [True, True, False, True, True],
         [False, True, True, False, True],
         [False, False, True, True, False]]
    )

    adjacency = add_two_hop_connections(direct)

    np.testing.assert_array_equal(adjacency, expected)
    np.testing.assert_array_equal(direct, original)


def test_two_hop_connections_do_not_join_disconnected_groups():
    direct = np.zeros((5, 5), dtype=bool)
    for i, j in [(0, 1), (1, 2), (3, 4)]:
        direct[i, j] = direct[j, i] = True
    expected = direct.copy()
    expected[0, 2] = expected[2, 0] = True

    np.testing.assert_array_equal(add_two_hop_connections(direct), expected)


def test_two_hop_connections_handle_many_common_neighbors():
    """256 条两步路径仍应判定为连接，避免小整数计数溢出。"""
    direct = np.zeros((258, 258), dtype=bool)
    direct[:2, 2:] = True
    direct[2:, :2] = True

    adjacency = add_two_hop_connections(direct)

    assert adjacency[0, 1]
    np.testing.assert_array_equal(adjacency, adjacency.T)
    assert not np.any(np.diag(adjacency))


def test_two_hop_connections_on_an_edgeless_graph():
    direct = np.zeros((3, 3), dtype=bool)

    np.testing.assert_array_equal(add_two_hop_connections(direct), direct)


# 3. 边界节点提取与全连接

def test_boundary_nodes_form_a_clique_and_preserve_existing_connections():
    labels = np.arange(9).reshape(3, 3)
    adjacency = np.zeros((9, 9), dtype=bool)
    adjacency[1, 4] = adjacency[4, 1] = True
    original_center_connections = adjacency[4].copy()

    connected, sides = add_boundary_connections(labels, adjacency)

    assert set(sides) == {"top", "bottom", "left", "right"}
    np.testing.assert_array_equal(sides["top"], [0, 1, 2])
    np.testing.assert_array_equal(sides["bottom"], [6, 7, 8])
    np.testing.assert_array_equal(sides["left"], [0, 3, 6])
    np.testing.assert_array_equal(sides["right"], [2, 5, 8])

    boundary = [0, 1, 2, 3, 5, 6, 7, 8]
    for i in boundary:
        for j in boundary:
            assert connected[i, j] == (i != j)

    # 内部节点 4 保留原来的连接，不因边界全连接而新增其他边。
    np.testing.assert_array_equal(connected[4], original_center_connections)
    np.testing.assert_array_equal(connected, connected.T)
    assert not np.any(np.diag(connected))
    assert np.count_nonzero(connected) == 8 * 7 + 2


def test_boundary_nodes_are_unique_per_side_and_corners_can_belong_to_two_sides():
    labels = np.array([[0, 0, 1, 1], [2, 2, 3, 3]])

    adjacency, sides = add_boundary_connections(
        labels, np.zeros((4, 4), dtype=bool)
    )

    np.testing.assert_array_equal(sides["top"], [0, 1])
    np.testing.assert_array_equal(sides["bottom"], [2, 3])
    np.testing.assert_array_equal(sides["left"], [0, 2])
    np.testing.assert_array_equal(sides["right"], [1, 3])
    np.testing.assert_array_equal(adjacency, ~np.eye(4, dtype=bool))


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1)])
def test_boundary_connections_on_small_and_thin_images(shape):
    labels = np.arange(np.prod(shape)).reshape(shape)
    n_nodes = labels.size

    adjacency, sides = add_boundary_connections(
        labels, np.zeros((n_nodes, n_nodes), dtype=bool)
    )

    # 单行、单列图像的所有节点都是图像边界节点。
    np.testing.assert_array_equal(adjacency, ~np.eye(n_nodes, dtype=bool))
    np.testing.assert_array_equal(sides["top"], np.unique(labels[0, :]))
    np.testing.assert_array_equal(sides["left"], np.unique(labels[:, 0]))


# 4. 边权计算

@pytest.mark.parametrize("sigma_square", [0.1, 0.5, 2.0])
def test_weights_use_normalized_euclidean_distance_and_sigma_square(sigma_square):
    """固定距离为 5、12、15，归一化后应为 0、0.7、1。"""
    mean_lab = np.array(
        [[50.0, 0.0, 0.0],
         [53.0, 4.0, 0.0],
         [50.0, 0.0, 12.0],
         [59.0, 12.0, 0.0]]
    )
    adjacency = np.array(
        [[False, True, True, True],
         [True, False, False, False],
         [True, False, False, False],
         [True, False, False, False]]
    )
    original_colors = mean_lab.copy()
    original_adjacency = adjacency.copy()

    weights = compute_weights(adjacency, mean_lab, sigma_square=sigma_square)

    expected = np.zeros((4, 4), dtype=np.float64)
    expected[0, 1] = expected[1, 0] = 1.0
    expected[0, 2] = expected[2, 0] = np.exp(-0.7 / sigma_square)
    expected[0, 3] = expected[3, 0] = np.exp(-1.0 / sigma_square)
    np.testing.assert_allclose(weights, expected, rtol=1e-12, atol=0)
    assert weights.dtype == np.float64
    assert np.all(np.isfinite(weights))
    assert np.all((weights >= 0) & (weights <= 1))
    np.testing.assert_array_equal(mean_lab, original_colors)
    np.testing.assert_array_equal(adjacency, original_adjacency)


def test_unconnected_color_outliers_do_not_change_weight_normalization():
    mean_lab = np.array([[50.0, 0, 0], [55.0, 0, 0], [65.0, 0, 0]])
    adjacency = np.array(
        [[False, True, True], [True, False, False], [True, False, False]]
    )
    weights = compute_weights(adjacency, mean_lab, sigma_square=0.5)

    extended_colors = np.vstack([mean_lab, [10000.0, 10000.0, 10000.0]])
    extended_adjacency = np.zeros((4, 4), dtype=bool)
    extended_adjacency[:3, :3] = adjacency
    extended_weights = compute_weights(
        extended_adjacency, extended_colors, sigma_square=0.5
    )

    np.testing.assert_allclose(extended_weights[:3, :3], weights)
    np.testing.assert_array_equal(extended_weights[3], 0)
    np.testing.assert_array_equal(extended_weights[:, 3], 0)


@pytest.mark.parametrize("distance", [0.0, 5.0])
def test_equal_edge_distances_follow_the_unit_weight_convention(distance):
    """覆盖全零距离，以及所有距离相等但不为零的分支。"""
    mean_lab = np.array([[50.0, 0, 0], [50.0 + distance, 0, 0]])
    adjacency = np.array([[False, True], [True, False]])

    weights = compute_weights(adjacency, mean_lab, sigma_square=0.1)

    np.testing.assert_array_equal(weights, [[0.0, 1.0], [1.0, 0.0]])


def test_larger_sigma_square_slows_weight_decay():
    mean_lab = np.array([[50.0, 0, 0], [55.0, 0, 0], [65.0, 0, 0]])
    adjacency = np.array(
        [[False, True, True], [True, False, False], [True, False, False]]
    )

    small = compute_weights(adjacency, mean_lab, sigma_square=0.1)
    large = compute_weights(adjacency, mean_lab, sigma_square=0.5)

    assert 0 < small[0, 2] < large[0, 2] < 1
    assert small[0, 1] == large[0, 1] == 1


def test_edgeless_graph_returns_zero_weights():
    """回归：没有连接时应该直接返回零矩阵，不能对空距离数组求最值。"""
    adjacency = np.zeros((1, 1), dtype=bool)
    mean_lab = np.array([[50.0, 0.0, 0.0]])

    weights = compute_weights(adjacency, mean_lab, sigma_square=0.1)

    np.testing.assert_array_equal(weights, [[0.0]])


# 5. 完整构图与 SLIC 接口

def test_build_graph_uses_spatial_two_hops_before_boundary_connections():
    """用规则网格的曼哈顿距离独立判断预期连接，不调用待测子函数。"""
    labels = np.arange(49).reshape(7, 7)
    mean_lab = np.column_stack([np.arange(49), np.zeros((49, 2))])
    original_labels = labels.copy()
    original_colors = mean_lab.copy()
    expected = np.zeros((49, 49), dtype=bool)

    for i in range(49):
        yi, xi = divmod(i, 7)
        boundary_i = yi in (0, 6) or xi in (0, 6)
        for j in range(49):
            yj, xj = divmod(j, 7)
            boundary_j = yj in (0, 6) or xj in (0, 6)
            distance = abs(yi - yj) + abs(xi - xj)
            expected[i, j] = i != j and (
                distance <= 2 or (boundary_i and boundary_j)
            )

    adjacency, weights, sides = build_graph(labels, mean_lab, sigma_square=0.2)

    np.testing.assert_array_equal(adjacency, expected)
    # 先连接边界再计算二阶邻接，会错误地把节点 8 和远处边界节点 13 连接。
    assert not adjacency[8, 13]
    assert weights[8, 13] == 0
    assert adjacency[0, 48]
    # 已连接节点的颜色距离最小为 1、最大为 48。
    assert weights[0, 1] == pytest.approx(1.0)
    assert weights[0, 48] == pytest.approx(np.exp(-1.0 / 0.2))
    assert set(sides) == {"top", "bottom", "left", "right"}
    np.testing.assert_array_equal(labels, original_labels)
    np.testing.assert_array_equal(mean_lab, original_colors)


@pytest.mark.parametrize("image_kind", ["uniform", "two_color", "gradient"])
def test_slic_outputs_connect_to_the_public_graph_interface(image_kind):
    """真实调用 SLIC，检查最终标签、区域均值和图节点编号能直接衔接。"""
    if image_kind == "uniform":
        image = np.full((12, 18, 3), [40, 120, 180], dtype=np.uint8)
    elif image_kind == "two_color":
        image = np.zeros((12, 18, 3), dtype=np.uint8)
        image[:, :9] = [220, 40, 30]
        image[:, 9:] = [30, 80, 200]
    else:
        yy, xx = np.indices((12, 18))
        image = np.stack(
            ((17 * yy + 13 * xx) % 256,
             (31 * yy + 7 * xx) % 256,
             (5 * yy + 29 * xx) % 256),
            axis=-1,
        ).astype(np.uint8)

    centers, labels, sizes = slic(image, n_segments=6, max_iter=3)
    original_labels = labels.copy()
    original_centers = centers.copy()

    # 不传 sigma_square，同时验证对外接口的默认参数可以使用。
    result = build_graph(labels, centers[:, :3])

    assert isinstance(result, tuple)
    assert len(result) == 3
    adjacency, weights, sides = result
    n_nodes = len(sizes)
    assert adjacency.shape == weights.shape == (n_nodes, n_nodes)
    assert adjacency.dtype == np.bool_
    assert weights.dtype == np.float64
    np.testing.assert_array_equal(np.unique(labels), np.arange(n_nodes))
    assert sizes.sum() == image.shape[0] * image.shape[1]
    np.testing.assert_array_equal(adjacency, adjacency.T)
    np.testing.assert_array_equal(weights, weights.T)
    assert not np.any(np.diag(adjacency))
    np.testing.assert_array_equal(np.diag(weights), 0)
    np.testing.assert_array_equal(weights[~adjacency], 0)
    assert np.all(np.isfinite(weights))
    assert np.all((weights[adjacency] > 0) & (weights[adjacency] <= 1))
    assert np.all(weights.sum(axis=1) > 0)
    assert set(sides) == {"top", "bottom", "left", "right"}
    np.testing.assert_array_equal(sides["top"], np.unique(labels[0, :]))
    np.testing.assert_array_equal(sides["bottom"], np.unique(labels[-1, :]))
    np.testing.assert_array_equal(sides["left"], np.unique(labels[:, 0]))
    np.testing.assert_array_equal(sides["right"], np.unique(labels[:, -1]))

    # 从像素接触关系直接核对：每一对直接相邻区域都应出现在最终图中。
    for first, second in (
        (labels[:, :-1], labels[:, 1:]),
        (labels[:-1, :], labels[1:, :]),
    ):
        different = first != second
        assert np.all(adjacency[first[different], second[different]])

    np.testing.assert_array_equal(labels, original_labels)
    np.testing.assert_array_equal(centers, original_centers)


def test_single_slic_region_can_build_a_graph_without_edges():
    """回归：一个超像素也应能通过完整接口，返回无自环的单节点图。"""
    image = np.full((6, 8, 3), 128, dtype=np.uint8)
    centers, labels, sizes = slic(image, n_segments=1, max_iter=2)

    adjacency, weights, sides = build_graph(labels, centers[:, :3])

    np.testing.assert_array_equal(sizes, [48])
    np.testing.assert_array_equal(adjacency, [[False]])
    np.testing.assert_array_equal(weights, [[0.0]])
    for nodes in sides.values():
        np.testing.assert_array_equal(nodes, [0])
