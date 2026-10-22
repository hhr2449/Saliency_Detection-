"""图像输入规范、图片读取和 RGB 到 Lab 转换测试。"""

import numpy as np
from PIL import Image, UnidentifiedImageError
import pytest

from slic.image import load_rgb, prepare_rgb, rgb_to_lab


def test_prepare_rgb_scales_uint8():
    image = np.array([[[0, 128, 255]]], dtype=np.uint8)
    original = image.copy()

    result = prepare_rgb(image)

    assert result.dtype == np.float32
    assert result.shape == image.shape
    np.testing.assert_allclose(result, [[[0, 128 / 255, 1]]], atol=1e-7)
    np.testing.assert_array_equal(image, original)
    assert not np.shares_memory(result, image)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_prepare_rgb_preserves_float_values(dtype):
    image = np.array([[[0, 0.25, 1], [0.5, 0.75, 0]]], dtype=dtype)
    original = image.copy()

    result = prepare_rgb(image)

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, image, atol=1e-7)
    np.testing.assert_array_equal(image, original)
    assert not np.shares_memory(result, image)


def test_prepare_rgb_rejects_non_array():
    with pytest.raises(TypeError):
        prepare_rgb([[[0, 0, 0]]])


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 1), (2, 3, 4), (0, 3, 3), (2, 0, 3)])
def test_prepare_rgb_rejects_invalid_shape(shape):
    with pytest.raises(ValueError):
        prepare_rgb(np.zeros(shape, dtype=np.float32))


@pytest.mark.parametrize("dtype", [np.int16, np.uint16, np.bool_, np.complex64])
def test_prepare_rgb_rejects_unsupported_dtype(dtype):
    with pytest.raises(TypeError):
        prepare_rgb(np.zeros((2, 2, 3), dtype=dtype))


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_prepare_rgb_rejects_invalid_float_values(value):
    image = np.zeros((2, 2, 3), dtype=np.float64)
    image[0, 0, 0] = value

    with pytest.raises(ValueError):
        prepare_rgb(image)


def test_load_rgb_reads_png_without_changing_channel_order(tmp_path):
    image = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [12, 128, 240]]],
        dtype=np.uint8,
    )
    image_path = tmp_path / "rgb.png"
    Image.fromarray(image).save(image_path)

    result = load_rgb(image_path)

    assert result.shape == (2, 2, 3)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, image.astype(np.float32) / 255, atol=1e-7)


def test_load_rgb_converts_grayscale_to_three_channels(tmp_path):
    gray = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    image_path = tmp_path / "gray.png"
    Image.fromarray(gray).save(image_path)

    result = load_rgb(image_path)

    expected = np.repeat(gray[..., None], 3, axis=2).astype(np.float32) / 255
    np.testing.assert_allclose(result, expected, atol=1e-7)


def test_load_rgb_discards_alpha_channel(tmp_path):
    rgba = np.array([[[10, 20, 30, 0], [40, 50, 60, 255]]], dtype=np.uint8)
    image_path = tmp_path / "rgba.png"
    Image.fromarray(rgba).save(image_path)

    result = load_rgb(image_path)

    np.testing.assert_allclose(result, rgba[..., :3].astype(np.float32) / 255, atol=1e-7)


def test_load_rgb_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rgb(tmp_path / "missing.png")


def test_load_rgb_rejects_invalid_image_file(tmp_path):
    image_path = tmp_path / "broken.png"
    image_path.write_bytes(b"not an image")

    with pytest.raises(UnidentifiedImageError):
        load_rgb(image_path)


def test_rgb_to_lab_known_black_and_white():
    image = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
    original = image.copy()

    lab = rgb_to_lab(image)

    assert lab.shape == image.shape
    assert lab.dtype == np.float32
    assert np.all(np.isfinite(lab))
    np.testing.assert_allclose(lab[0, 0], [0, 0, 0], atol=0.02)
    np.testing.assert_allclose(lab[0, 1], [100, 0, 0], atol=0.02)
    np.testing.assert_array_equal(image, original)


def test_rgb_to_lab_primary_colors_have_expected_color_directions():
    image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)

    red, green, blue = rgb_to_lab(image)[0]

    assert red[1] > 0 and red[2] > 0
    assert green[1] < 0 and green[2] > 0
    assert blue[2] < 0
    assert green[0] > red[0] > blue[0]


def test_rgb_to_lab_uint8_and_float_inputs_agree():
    image = np.array([[[20, 80, 160], [255, 128, 0]]], dtype=np.uint8)
    float_image = image.astype(np.float32) / 255

    np.testing.assert_allclose(rgb_to_lab(image), rgb_to_lab(float_image), atol=1e-5)


def test_rgb_to_lab_reuses_input_validation():
    with pytest.raises(ValueError):
        rgb_to_lab(np.full((2, 2, 3), 1.1, dtype=np.float32))
