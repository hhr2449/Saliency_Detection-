"""测试公共配置：导入 src 包，并提供独立的四连通性检查。"""

from collections import deque
from pathlib import Path
import sys

import numpy as np
import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def assert_four_connected():
    """通过独立 BFS 检查每个标签，避免使用待测的连通块函数自检。"""
    def check(labels):
        H, W = labels.shape

        for label in np.unique(labels):
            ys, xs = np.where(labels == label)
            start = (int(ys[0]), int(xs[0]))
            visited = {start}
            queue = deque([start])

            while queue:
                y, x = queue.popleft()

                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if not (0 <= ny < H and 0 <= nx < W):
                        continue
                    if labels[ny, nx] != label or (ny, nx) in visited:
                        continue

                    visited.add((ny, nx))
                    queue.append((ny, nx))

            assert len(visited) == len(ys), f"标签 {label} 包含不连通的部分"

    return check
