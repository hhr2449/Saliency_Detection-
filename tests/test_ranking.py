"""非归一化关联矩阵、查询计算与退化输入测试。

预期矩阵使用可手算结果，参数异常沿用当前接口的异常类型。
"""

import numpy as np
import pytest

from ranking import compute_ranking_matrix, rank_queries


def test_two_node_ranking_matrix_matches_the_analytic_solution():
    """D=I、alpha=1/2，求逆并清零后两个非对角元素均为 2/3。"""
    weights = np.array([[0, 1], [1, 0]], dtype=np.int64)
    original = weights.copy()

    matrix = compute_ranking_matrix(weights, alpha=0.5)

    expected = np.array([[0.0, 2 / 3], [2 / 3, 0.0]])
    np.testing.assert_allclose(matrix, expected, rtol=1e-12, atol=0)
    assert matrix.dtype == np.float64
    np.testing.assert_array_equal(weights, original)


def test_weighted_degrees_are_used_in_the_ranking_matrix():
    """边权为 2、1 的三节点链，D 应为 diag(2, 3, 1)。"""
    weights = np.array([[0.0, 2, 0], [2, 0, 1], [0, 1, 0]])
    expected = np.array(
        [[0.0, 2 / 9, 1 / 9],
         [2 / 9, 0.0, 2 / 9],
         [1 / 9, 2 / 9, 0.0]]
    )

    matrix = compute_ranking_matrix(weights, alpha=0.5)

    np.testing.assert_allclose(matrix, expected, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(np.diag(matrix), 0)
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-12, atol=0)
    assert np.all(np.isfinite(matrix))


def test_default_alpha_is_099():
    weights = np.array([[0.0, 1], [1, 0]])

    matrix = compute_ranking_matrix(weights)

    expected_off_diagonal = 0.99 / (1 - 0.99**2)
    np.testing.assert_allclose(
        matrix,
        [[0.0, expected_off_diagonal], [expected_off_diagonal, 0.0]],
        rtol=1e-12,
        atol=0,
    )


def test_isolated_node_and_disconnected_components_keep_their_node_ids():
    """节点 1 孤立；0—2 和 3—4 两个分量应分别计算，跨分量关联为零。"""
    weights = np.zeros((5, 5), dtype=np.float64)
    weights[0, 2] = weights[2, 0] = 1
    weights[3, 4] = weights[4, 3] = 2
    expected = np.zeros((5, 5), dtype=np.float64)
    expected[0, 2] = expected[2, 0] = 2 / 3
    expected[3, 4] = expected[4, 3] = 1 / 3

    matrix = compute_ranking_matrix(weights, alpha=0.5)

    np.testing.assert_allclose(matrix, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("n_nodes", [0, 1, 4])
def test_edgeless_graph_returns_a_zero_ranking_matrix(n_nodes):
    weights = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    matrix = compute_ranking_matrix(weights)

    assert matrix.shape == weights.shape
    assert matrix.dtype == np.float64
    np.testing.assert_array_equal(matrix, np.zeros_like(weights))


def test_scaling_all_weights_scales_the_unnormalized_matrix_inversely():
    weights = np.array([[0.0, 2, 0], [2, 0, 1], [0, 1, 0]])

    original = compute_ranking_matrix(weights, alpha=0.5)
    scaled = compute_ranking_matrix(5 * weights, alpha=0.5)

    np.testing.assert_allclose(scaled, original / 5, rtol=1e-12, atol=0)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1, np.nan, np.inf])
def test_ranking_matrix_rejects_invalid_alpha(alpha):
    with pytest.raises(AssertionError, match="alpha"):
        compute_ranking_matrix(np.array([[0.0, 1], [1, 0]]), alpha=alpha)


@pytest.mark.parametrize(
    "weights",
    [np.array(1.0), np.ones(3), np.ones((2, 3)), np.ones((2, 2, 1))],
)
def test_ranking_matrix_rejects_non_square_inputs(weights):
    with pytest.raises(AssertionError, match="方阵"):
        compute_ranking_matrix(weights)


@pytest.mark.parametrize(
    "query_vector, expected",
    [
        ([1, 0, 0], [0, 2, 1]),
        ([0, 1, 1], [3, 3, 3]),
        ([0, 0, 0], [0, 0, 0]),
        ([1, 1, 1], [3, 5, 4]),
    ],
)
def test_rank_queries_returns_raw_scores_for_single_and_multiple_queries(
    query_vector, expected
):
    matrix = np.array([[0, 2, 1], [2, 0, 3], [1, 3, 0]], dtype=np.int64)
    query_vector = np.asarray(query_vector, dtype=np.int64)
    original_matrix = matrix.copy()
    original_query = query_vector.copy()

    scores = rank_queries(matrix, query_vector)

    assert scores.shape == (3,)
    assert scores.dtype == np.float64
    np.testing.assert_array_equal(scores, expected)
    np.testing.assert_array_equal(matrix, original_matrix)
    np.testing.assert_array_equal(query_vector, original_query)


def test_ranking_matrix_can_be_reused_for_different_queries():
    weights = np.array([[0.0, 2, 0], [2, 0, 1], [0, 1, 0]])
    matrix = compute_ranking_matrix(weights, alpha=0.5)
    original = matrix.copy()

    first = rank_queries(matrix, np.array([1, 0, 0]))
    second = rank_queries(matrix, np.array([0, 0, 1]))
    combined = rank_queries(matrix, np.array([1, 0, 1]))

    np.testing.assert_allclose(first, [0.0, 2 / 9, 1 / 9], rtol=1e-12, atol=0)
    np.testing.assert_allclose(second, [1 / 9, 2 / 9, 0.0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(combined, first + second, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(matrix, original)


@pytest.mark.parametrize("n_nodes", [0, 1, 4])
def test_rank_queries_on_a_zero_matrix(n_nodes):
    matrix = np.zeros((n_nodes, n_nodes))

    scores = rank_queries(matrix, np.ones(n_nodes))

    assert scores.shape == (n_nodes,)
    np.testing.assert_array_equal(scores, np.zeros(n_nodes))


@pytest.mark.parametrize(
    "matrix", [np.ones(3), np.ones((2, 3)), np.ones((2, 2, 1))]
)
def test_rank_queries_rejects_non_square_matrices(matrix):
    with pytest.raises(ValueError, match="方阵"):
        rank_queries(matrix, np.ones(2))


@pytest.mark.parametrize(
    "query_vector",
    [np.array([]), np.ones(2), np.ones(4), np.ones((3, 1)), np.ones((1, 3))],
)
def test_rank_queries_rejects_query_vectors_with_wrong_shapes(query_vector):
    with pytest.raises(ValueError, match="query_vector"):
        rank_queries(np.zeros((3, 3)), query_vector)
