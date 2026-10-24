"""调用 SLIC 接口，输出三种图像、三联展示图及 metadata。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from slic.core import slic
from slic.image import load_rgb
from visualization.superpixels import mean_color_image, overlay_boundaries, plot_comparison


def build_parser() -> argparse.ArgumentParser:
    """读取图像路径、图名、保存路径和可选的 SLIC 参数。"""
    parser = argparse.ArgumentParser(description="SLIC 分割和可视化演示")
    parser.add_argument("image", type=Path, help="输入图像路径")
    parser.add_argument(
        "--name", "--image-name", dest="image_name",
        help="图名，默认使用源文件名（不含扩展名）",
    )
    parser.add_argument(
        "--save-dir", "--output-dir", dest="save_dir", type=Path,
        help="自定义结果目录，指定后覆盖默认路径；省略时保存到项目根目录的 outputs/slic/<图名>/",
    )
    parser.add_argument("--n-segments", type=int, default=200, help="期望超像素数量")
    parser.add_argument("--compactness", type=float, default=10.0, help="紧致度参数")
    parser.add_argument("--max-iter", type=int, default=10, help="聚类迭代次数")
    parser.add_argument("--min-size-factor", type=float, default=0.25, help="小区域面积阈值比例")
    parser.add_argument("--show", action="store_true", help="保存后显示三联展示图窗口")
    return parser


def save_rgb(image: np.ndarray, image_path: Path) -> None:
    """保存 [0, 1] 的 RGB 图像，四舍五入转换为 uint8 PNG。"""
    pixels = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(image_path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    image_name = args.image_name or args.image.stem

    if Path(image_name).name != image_name or image_name in (".", ".."):
        parser.error("图名不能包含目录，请使用 --save-dir 指定保存路径。")

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = PROJECT_ROOT / "outputs" / "slic" / image_name

    # 默认仅保存图片，支持没有图形界面的终端环境。
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")

    image = load_rgb(args.image)
    start = perf_counter()
    centers, labels, sizes = slic(
        image,
        n_segments=args.n_segments,
        compactness=args.compactness,
        max_iter=args.max_iter,
        min_size_factor=args.min_size_factor,
    )
    slic_seconds = perf_counter() - start

    start = perf_counter()
    boundaries = overlay_boundaries(image, labels)
    mean_colors = mean_color_image(image, labels)
    fig = plot_comparison(
        image,
        boundaries,
        mean_colors,
        title=f"{image_name} | K={args.n_segments}, N={len(sizes)}, m={args.compactness:g}",
    )
    visualization_seconds = perf_counter() - start

    save_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "original": "original.png",
        "boundaries": "boundaries.png",
        "mean_colors": "mean_colors.png",
        "comparison": "comparison.png",
        "metadata": "metadata.json",
    }
    save_rgb(image, save_dir / files["original"])
    save_rgb(boundaries, save_dir / files["boundaries"])
    save_rgb(mean_colors, save_dir / files["mean_colors"])
    fig.savefig(save_dir / files["comparison"], dpi=160)

    H, W = image.shape[:2]
    metadata = {
        "image": {
            "name": image_name,
            "source_path": str(args.image.resolve()),
            "height": H,
            "width": W,
            "channels": image.shape[2],
        },
        "parameters": {
            "n_segments": args.n_segments,
            "compactness": args.compactness,
            "max_iter": args.max_iter,
            "min_size_factor": args.min_size_factor,
        },
        "result": {
            "n_segments": len(sizes),
            "centers_shape": list(centers.shape),
            "labels_shape": list(labels.shape),
            "area_min": int(sizes.min()),
            "area_max": int(sizes.max()),
            "area_mean": float(sizes.mean()),
            "area_total": int(sizes.sum()),
        },
        "timing_seconds": {
            "slic": slic_seconds,
            "visualization": visualization_seconds,
        },
        "files": files,
    }
    with (save_dir / files["metadata"]).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"图名：{image_name}")
    print(f"目标超像素数：{args.n_segments}，实际超像素数：{len(sizes)}")
    print(f"SLIC 耗时：{slic_seconds:.3f} 秒")
    print(f"保存目录：{save_dir.resolve()}")

    import matplotlib.pyplot as plt
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
