"""背景融合、前景查询、归一化以及完整显著性流程的功能与边界测试。"""

import numpy as np
import pytest

import saliency.background as background_module
from graph import build_graph
from ranking import compute_ranking_matrix, rank_queries
from saliency import background_query, foreground_query
from saliency.utils import normalize_scores
from slic.core import slic


@pytest.fixture
def background_matrix():
    """四个边界节点 0—3，两个内部节点 4、5。"""
    return np.array(
        [[0.0, 2, 2, 2, 1, 0.5],
         [2, 0, 2, 2, 1, 0.5],
         [2, 2, 0, 2, 1, 0.5],
         [2, 2, 2, 0, 1, 0.5],
         [1, 1, 1, 1, 0, 0.25],
         [0.5, 0.5, 0.5, 0.5, 0.25, 0]]
    )


@pytest.fixture
def four_sides():
    return {
        "top": np.array([0]),
        "bottom": np.array([1]),
        "left": np.array([2]),
        "right": np.array([3]),
    }


@pytest.fixture
def foreground_matrix():
    return np.array(
        [[0.0, 1, 3, 2],
         [1, 0, 4, 5],
         [3, 4, 0, 6],
         [2, 5, 6, 0]]
    )


# 1. 分数归一化

@pytest.mark.parametrize(
    "scores, expected",
    [
        ([2, 4, 6], [0.0, 0.5, 1.0]),
        ([-3, -1, 1], [0.0, 0.5, 1.0]),
        ([0, 0, 0], [0.0, 0.0, 0.0]),
        ([7, 7, 7], [0.0, 0.0, 0.0]),
        ([5], [0.0]),
    ],
)
def test_normalize_scores_uses_min_max_and_handles_constant_scores(scores, expected):
    scores = np.asarray(scores)
    original = scores.copy()

    normalized = normalize_scores(scores)

    assert normalized.shape == scores.shape
    assert normalized.dtype == np.float64
    np.testing.assert_allclose(normalized, expected, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(scores, original)


def test_normalization_preserves_small_but_nonzero_contrast():
    # 用可精确表示的二进制间隔，避免测试自身的十进制舍入误差。
    delta = 2.0**-40
    scores = np.array([1.0, 1.0 + delta, 1.0 + 2 * delta])

    np.testing.assert_allclose(normalize_scores(scores), [0.0, 0.5, 1.0])


def test_normalization_rejects_empty_scores():
    with pytest.raises(ValueError):
        normalize_scores(np.array([]))


# 2. 背景查询

def test_background_query_normalizes_each_side_and_multiplies_scores(
    background_matrix, four_sides
):
    original_matrix = background_matrix.copy()
    original_sides = {side: nodes.copy() for side, nodes in four_sides.items()}

    scores = background_query(background_matrix, four_sides)

    # 每侧给内部节点 4、5 的显著分数为 1/2、3/4。
    # 四侧乘积为 1/16、81/256，再归一化得到 16/81、1。
    np.testing.assert_allclose(scores, [0, 0, 0, 0, 16 / 81, 1], rtol=1e-12, atol=0)
    assert scores.dtype == np.float64
    np.testing.assert_array_equal(background_matrix, original_matrix)
    for side in four_sides:
        np.testing.assert_array_equal(four_sides[side], original_sides[side])


def test_background_query_builds_four_separate_query_vectors(
    background_matrix, four_sides, monkeypatch
):
    calls = []

    def record_query(matrix, query_vector):
        np.testing.assert_array_equal(matrix, background_matrix)
        calls.append(query_vector.copy())
        return rank_queries(matrix, query_vector)

    monkeypatch.setattr(background_module, "rank_queries", record_query)

    background_query(background_matrix, four_sides)

    assert len(calls) == 4
    expected = np.array(
        [[1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0]]
    )
    np.testing.assert_array_equal(np.stack(calls), expected)


def test_repeated_nodes_and_shared_corners_do_not_change_background_results(
    background_matrix,
):
    sides = {
        "top": np.array([0, 1]),
        "bottom": np.array([2, 3]),
        "left": np.array([0, 2]),
        "right": np.array([1, 3]),
    }
    repeated = {side: np.repeat(nodes, 2) for side, nodes in sides.items()}

    expected = background_query(background_matrix, sides)
    actual = background_query(background_matrix, repeated)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


def test_empty_side_contributes_a_neutral_factor(background_matrix, four_sides):
    four_sides["right"] = np.array([], dtype=np.int64)

    scores = background_query(background_matrix, four_sides)

    # 剩余三侧乘积为 1/8、27/64；空查询的一侧贡献乘数 1。
    np.testing.assert_allclose(scores, [0, 0, 0, 0, 8 / 27, 1], rtol=1e-12, atol=0)


def test_all_empty_boundary_queries_return_zero_scores(background_matrix):
    sides = {
        side: np.array([], dtype=np.int64)
        for side in ("top", "bottom", "left", "right")
    }

    np.testing.assert_array_equal(background_query(background_matrix, sides), np.zeros(6))


@pytest.mark.parametrize("n_nodes", [1, 4])
def test_background_query_on_an_edgeless_graph(n_nodes):
    matrix = np.zeros((n_nodes, n_nodes))
    sides = {side: np.array([0]) for side in ("top", "bottom", "left", "right")}

    np.testing.assert_array_equal(background_query(matrix, sides), np.zeros(n_nodes))


def test_background_query_requires_all_four_side_keys(background_matrix, four_sides):
    del four_sides["right"]

    with pytest.raises(KeyError, match="right"):
        background_query(background_matrix, four_sides)


def test_background_query_rejects_out_of_range_node_ids(background_matrix, four_sides):
    four_sides["top"] = np.array([6])

    with pytest.raises(IndexError):
        background_query(background_matrix, four_sides)


def test_background_query_rejects_a_non_square_matrix(four_sides):
    with pytest.raises(ValueError, match="方阵"):
        background_query(np.ones((6, 5)), four_sides)


# 3. 前景查询

def test_foreground_query_selects_high_scores_and_normalizes_without_inverting(
    foreground_matrix,
):
    initial_scores = np.array([0.0, 0.2, 0.8, 1.0])
    original_scores = initial_scores.copy()
    original_matrix = foreground_matrix.copy()

    scores = foreground_query(foreground_matrix, initial_scores)

    # 阈值为 0.5，查询节点为 2、3，原始关联分数为 [5, 9, 6, 6]。
    np.testing.assert_allclose(scores, [0.0, 1.0, 0.25, 0.25], rtol=1e-12, atol=0)
    assert scores.dtype == np.float64
    np.testing.assert_array_equal(initial_scores, original_scores)
    np.testing.assert_array_equal(foreground_matrix, original_matrix)


def test_foreground_query_uses_the_arithmetic_mean_threshold(foreground_matrix):
    initial_scores = np.array([0.0, 0.1, 0.2, 1.0])

    scores = foreground_query(foreground_matrix, initial_scores)

    # 均值为 0.325，只有节点 3 入选，原始关联分数为 [2, 5, 6, 0]。
    np.testing.assert_allclose(scores, [1 / 3, 5 / 6, 1.0, 0.0], rtol=1e-12, atol=0)


def test_foreground_query_includes_nodes_equal_to_the_mean(foreground_matrix):
    initial_scores = np.array([0.0, 0.5, 0.5, 1.0])

    scores = foreground_query(foreground_matrix, initial_scores)

    # 1、2、3 均入选，原始关联分数为 [6, 9, 10, 11]。
    np.testing.assert_allclose(scores, [0.0, 3 / 5, 4 / 5, 1.0], rtol=1e-12, atol=0)


@pytest.mark.parametrize("value", [0.0, 0.7])
def test_equal_initial_scores_select_all_nodes(foreground_matrix, value):
    initial_scores = np.full(4, value)

    scores = foreground_query(foreground_matrix, initial_scores)

    # 使用 >= 均值的约定，所有节点入选；行和为 [6, 10, 13, 13]。
    np.testing.assert_allclose(scores, [0.0, 4 / 7, 1.0, 1.0], rtol=1e-12, atol=0)


def test_constant_nonzero_foreground_rank_scores_normalize_to_zero():
    matrix = np.ones((3, 3)) - np.eye(3)

    scores = foreground_query(matrix, np.ones(3))

    np.testing.assert_array_equal(scores, np.zeros(3))


@pytest.mark.parametrize("n_nodes", [1, 4])
def test_foreground_query_on_an_edgeless_graph(n_nodes):
    scores = foreground_query(np.zeros((n_nodes, n_nodes)), np.arange(n_nodes))

    np.testing.assert_array_equal(scores, np.zeros(n_nodes))


@pytest.mark.parametrize(
    "initial_scores",
    [
        np.array([]),
        np.ones((1, 4)),
        np.array([0.0, np.nan, 0.5, 1.0]),
        np.array([0.0, np.inf, 0.5, 1.0]),
        np.array([0.0, -np.inf, 0.5, 1.0]),
    ],
)
def test_foreground_query_rejects_invalid_initial_scores(foreground_matrix, initial_scores):
    with pytest.raises(ValueError, match="initial_scores"):
        foreground_query(foreground_matrix, initial_scores)


def test_foreground_query_rejects_a_node_count_mismatch(foreground_matrix):
    with pytest.raises(ValueError, match="query_vector"):
        foreground_query(foreground_matrix, np.array([0.0, 1.0]))


# 4. SLIC、建图、排序、显著性查询的接口衔接

@pytest.mark.parametrize("image_kind", ["uniform", "two_color"])
def test_complete_pipeline_reuses_the_ranking_matrix_and_restores_pixel_scores(image_kind):
    image = np.full((12, 18, 3), [40, 120, 180], dtype=np.uint8)
    if image_kind == "two_color":
        image[:, :9] = [220, 40, 30]
        image[:, 9:] = [30, 80, 200]
    original_image = image.copy()
    centers, labels, sizes = slic(image, n_segments=6, max_iter=3)
    _, weights, sides = build_graph(labels, centers[:, :3])
    original_weights = weights.copy()
    matrix = compute_ranking_matrix(weights)
    original_matrix = matrix.copy()

    initial_scores = background_query(matrix, sides)
    final_scores = foreground_query(matrix, initial_scores)
    saliency_map = final_scores[labels]

    n_nodes = len(sizes)
    for scores in (initial_scores, final_scores):
        assert scores.shape == (n_nodes,)
        assert scores.dtype == np.float64
        assert np.all(np.isfinite(scores))
        assert np.all((scores >= 0) & (scores <= 1))
    assert saliency_map.shape == image.shape[:2]
    for k in range(n_nodes):
        np.testing.assert_array_equal(saliency_map[labels == k], final_scores[k])
    np.testing.assert_array_equal(matrix, original_matrix)
    np.testing.assert_array_equal(weights, original_weights)
    np.testing.assert_array_equal(image, original_image)


@pytest.mark.parametrize("shape", [(1, 1), (1, 8), (8, 1), (5, 7)])
def test_single_superpixel_pipeline_returns_finite_zero_scores(shape):
    image = np.full((*shape, 3), 128, dtype=np.uint8)
    centers, labels, _ = slic(image, n_segments=1, max_iter=2)
    _, weights, sides = build_graph(labels, centers[:, :3])

    matrix = compute_ranking_matrix(weights)
    initial_scores = background_query(matrix, sides)
    final_scores = foreground_query(matrix, initial_scores)

    np.testing.assert_array_equal(matrix, [[0.0]])
    np.testing.assert_array_equal(initial_scores, [0.0])
    np.testing.assert_array_equal(final_scores, [0.0])
    np.testing.assert_array_equal(final_scores[labels], np.zeros(shape))
