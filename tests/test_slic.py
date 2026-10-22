"""完整 RGB 分割流程及最终标签、中心、面积之间的一致性测试。"""

import numpy as np
import pytest

import slic.core as core
from slic.image import rgb_to_lab


def _assert_consistent_result(image, result, assert_four_connected):
    """从返回标签独立核对面积和特征，检查是否错误返回了合并前的中心。"""
    assert isinstance(result, tuple)
    assert len(result) == 3
    centers, labels, sizes = result
    N = len(sizes)

    assert labels.shape == image.shape[:2]
    assert centers.shape == (N, 5)
    assert np.issubdtype(labels.dtype, np.integer)
    assert np.issubdtype(sizes.dtype, np.integer)
    assert np.all(labels >= 0)
    assert np.all(sizes > 0)
    assert np.all(np.isfinite(centers))
    assert sizes.sum() == image.shape[0] * image.shape[1]
    np.testing.assert_array_equal(np.unique(labels), np.arange(N))
    np.testing.assert_array_equal(sizes, np.bincount(labels.ravel()))
    assert_four_connected(labels)

    lab = rgb_to_lab(image).astype(np.float64)
    for k in range(N):
        mask = labels == k
        ys, xs = np.where(mask)
        np.testing.assert_allclose(
            centers[k, :3], lab[mask].mean(axis=0), rtol=1e-6, atol=2e-5
        )
        np.testing.assert_allclose(centers[k, 3:], [ys.mean(), xs.mean()], atol=1e-12)


def test_slic_uniform_image_returns_consistent_regions(assert_four_connected):
    image = np.full((8, 12, 3), [40, 120, 180], dtype=np.uint8)
    original = image.copy()

    result = core.slic(image, n_segments=6, max_iter=4)

    _assert_consistent_result(image, result, assert_four_connected)
    np.testing.assert_array_equal(image, original)


def test_slic_preserves_a_strong_two_color_boundary(assert_four_connected):
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[:, :6] = [255, 0, 0]
    image[:, 6:] = [0, 0, 255]

    result = core.slic(image, n_segments=6, compactness=5, max_iter=4)

    _assert_consistent_result(image, result, assert_four_connected)
    _, labels, _ = result
    assert np.intersect1d(labels[:, :6], labels[:, 6:]).size == 0


@pytest.mark.parametrize("H, W, K", [(1, 1, 1), (1, 9, 3), (9, 1, 3), (7, 11, 8)])
def test_slic_handles_small_and_thin_images(H, W, K, assert_four_connected):
    yy, xx = np.indices((H, W))
    image = np.stack(
        ((17 * yy + 13 * xx) % 256, (31 * yy + 7 * xx) % 256, (5 * yy + 29 * xx) % 256),
        axis=-1,
    ).astype(np.uint8)

    result = core.slic(image, n_segments=K, max_iter=3)

    _assert_consistent_result(image, result, assert_four_connected)


@pytest.mark.parametrize("shape", [(5, 5), (1, 25), (25, 1)])
def test_slic_one_requested_segment_covers_the_whole_uniform_image(shape, assert_four_connected):
    image = np.full((*shape, 3), 128, dtype=np.uint8)

    result = core.slic(image, n_segments=1, max_iter=3)

    _assert_consistent_result(image, result, assert_four_connected)
    centers, labels, sizes = result
    assert centers.shape == (1, 5)
    np.testing.assert_array_equal(labels, np.zeros(shape, dtype=int))
    np.testing.assert_array_equal(sizes, [shape[0] * shape[1]])


def test_slic_is_deterministic_for_fixed_inputs():
    image = np.random.default_rng(42).integers(0, 256, size=(6, 8, 3), dtype=np.uint8)

    first = core.slic(image, n_segments=6, max_iter=3)
    second = core.slic(image, n_segments=6, max_iter=3)

    for first_array, second_array in zip(first, second):
        np.testing.assert_array_equal(first_array, second_array)


def test_slic_uint8_and_float_inputs_agree():
    image = np.random.default_rng(42).integers(0, 256, size=(6, 8, 3), dtype=np.uint8)

    integer_result = core.slic(image, n_segments=6, max_iter=3)
    float_result = core.slic(image.astype(np.float32) / 255, n_segments=6, max_iter=3)

    np.testing.assert_allclose(integer_result[0], float_result[0], atol=1e-5)
    np.testing.assert_array_equal(integer_result[1], float_result[1])
    np.testing.assert_array_equal(integer_result[2], float_result[2])


def test_final_statistics_are_recomputed_after_merging(monkeypatch, assert_four_connected):
    """固定聚类标签，确保触发小块合并，再检查最终统计。"""
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    image[1, 1] = [255, 0, 0]
    raw_labels = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int32)
    monkeypatch.setattr(core, "assign_pixels", lambda *args: raw_labels.copy())

    result = core.slic(image, n_segments=2, max_iter=1, min_size_factor=1)

    _assert_consistent_result(image, result, assert_four_connected)
    centers, labels, sizes = result
    assert centers.shape == (1, 5)
    np.testing.assert_array_equal(labels, np.zeros((3, 3), dtype=int))
    np.testing.assert_array_equal(sizes, [9])
    assert centers[0, 0] > 0  # 合并后的均值必须包含中心红色像素。


def test_merge_threshold_uses_actual_initial_center_count(monkeypatch, assert_four_connected):
    """目标 K=7、实际 K0=8 时，面积 6 的块应保留：阈值为 6 而非 7。"""
    image = np.zeros((12, 17, 3), dtype=np.uint8)
    raw_labels = np.zeros((12, 17), dtype=np.int32)
    raw_labels[-1, -6:] = 1
    monkeypatch.setattr(
        core,
        "initialize_centers",
        lambda lab, n_segments: (np.zeros((8, 5), dtype=np.float64), 2.0),
    )
    monkeypatch.setattr(core, "assign_pixels", lambda *args: raw_labels.copy())

    result = core.slic(image, n_segments=7, max_iter=1, min_size_factor=0.25)

    _assert_consistent_result(image, result, assert_four_connected)
    np.testing.assert_array_equal(np.sort(result[2]), [6, 198])


@pytest.mark.parametrize(
    "kwargs, expected_error",
    [
        ({"max_iter": 0}, ValueError),
        ({"max_iter": -1}, ValueError),
        ({"max_iter": 1.5}, TypeError),
        ({"max_iter": True}, TypeError),
        ({"min_size_factor": -0.1}, ValueError),
        ({"min_size_factor": np.nan}, ValueError),
        ({"min_size_factor": np.inf}, ValueError),
    ],
)
def test_slic_rejects_invalid_parameters(kwargs, expected_error):
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(expected_error):
        core.slic(image, n_segments=4, **kwargs)
