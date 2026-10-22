"""图像读取、输入检查和颜色空间转换。"""

import numpy as np
from PIL import Image

image_path = "/home/hhr/3_autumn/media/Saliency_Detection/images/"

def load_rgb(image_path: str | PathLike[str]) -> np.ndarray:
    """
    读取图片，返回范围为 [0, 1] 的 RGB float32 数组。
    返回形状为 (H, W, 3)。
    灰度图等格式统一转换为 RGB；透明图像直接移除 alpha 通道。
    文件不存在或格式无法识别时，保留 Pillow 的原始异常。
    """
    with Image.open(image_path) as image:
        rgb = np.array(image.convert("RGB"))

    return prepare_rgb(rgb)


def prepare_rgb(image: np.ndarray) -> np.ndarray:
    """检查并统一 RGB 图像。
    支持的输入：
        uint8 数组，范围为 [0, 255]；
        浮点数组，范围为 [0, 1]。
    返回：
        形状为 (H, W, 3) 的 float32 数组，范围为 [0, 1]。
        返回独立数组，不修改输入。
    """

    # uint8 的取值天然位于 [0, 255]，直接转换并缩放。
    if image.dtype == np.uint8:
        return image.astype(np.float32) / np.float32(255.0)

    if not np.issubdtype(image.dtype, np.floating):
        raise TypeError("图像仅支持 uint8 或浮点类型。")

    return image.astype(np.float32, copy=True)


