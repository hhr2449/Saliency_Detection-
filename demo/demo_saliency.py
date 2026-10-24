"""调用完整检测入口，保存显著性结果、对比图和 metadata"""

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

from pipeline import detect_saliency
from slic.image import load_rgb


def build_parser() -> argparse.ArgumentParser:
    """读取图像路径、图名、保存路径和算法参数"""
    parser = argparse.ArgumentParser(description="完整显著性检测和可视化演示")
    parser.add_argument("image", type=Path, help="输入图像路径")
    parser.add_argument(
        "--name", "--image-name", dest="image_name",
        help="图名，默认使用源文件名（不含扩展名）",
    )
    parser.add_argument(
        "--save-dir", "--output-dir", dest="save_dir", type=Path,
        help="自定义结果目录，指定后覆盖默认路径；省略时保存到项目根目录的 outputs/saliency/<图名>/",
    )
    parser.add_argument("--n-segments", type=int, default=200, help="期望超像素数量")
    parser.add_argument("--compactness", type=float, default=10.0, help="SLIC 紧致度参数")
    parser.add_argument("--max-iter", type=int, default=10, help="SLIC 聚类迭代次数")
    parser.add_argument("--min-size-factor", type=float, default=0.25, help="小区域面积阈值比例")
    parser.add_argument("--sigma-square", type=float, default=0.1, help="颜色边权参数 sigma²")
    parser.add_argument("--alpha", type=float, default=0.99, help="流形排序参数")
    parser.add_argument("--show", action="store_true", help="保存后显示两阶段对比图和四联图窗口")
    return parser


def save_image(image: np.ndarray, image_path: Path) -> None:
    """将 [0, 1] 的 RGB、灰度或二值图像保存为 uint8 PNG"""
    pixels = np.rint(image * 255).astype(np.uint8)
    Image.fromarray(pixels).save(image_path)


def main() -> None:
    args = build_parser().parse_args()
    image_name = args.image_name or args.image.stem
    save_dir = args.save_dir or PROJECT_ROOT / "outputs" / "saliency" / image_name

    # 先设置绘图后端，再导入可视化接口
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from visualization import (
        saliency_image,
        foreground_query_image,
        plot_stage_comparison,
        plot_saliency_comparison,
    )

    image = load_rgb(args.image)
    start = perf_counter()
    result = detect_saliency(
        image,
        n_segments=args.n_segments,
        compactness=args.compactness,
        max_iter=args.max_iter,
        min_size_factor=args.min_size_factor,
        sigma_square=args.sigma_square,
        alpha=args.alpha,
        return_details=True,
    )
    detection_seconds = perf_counter() - start

    labels = result["labels"]
    sizes = result["sizes"]
    initial_scores = result["initial_scores"]
    n_nodes = len(sizes)

    # 先生成三张结果图，再生成两阶段对比图和四联图
    start = perf_counter()
    initial_image = saliency_image(initial_scores, labels)
    query_image = foreground_query_image(initial_scores, labels)
    final_image = result["saliency_map"]
    title = f"{image_name} | N={n_nodes}, alpha={args.alpha:g}, sigma²={args.sigma_square:g}"
    stage_fig = plot_stage_comparison(initial_image, final_image, title=title)
    comparison_fig = plot_saliency_comparison(
        image, initial_image, query_image, final_image, title=title,
    )
    visualization_seconds = perf_counter() - start

    save_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "original": "original.png",
        "initial_saliency": "initial_saliency.png",
        "foreground_queries": "foreground_queries.png",
        "final_saliency": "final_saliency.png",
        "stage_comparison": "stage_comparison.png",
        "comparison": "comparison.png",
        "metadata": "metadata.json",
    }
    save_image(image, save_dir / files["original"])
    save_image(initial_image, save_dir / files["initial_saliency"])
    save_image(query_image, save_dir / files["foreground_queries"])
    save_image(final_image, save_dir / files["final_saliency"])
    stage_fig.savefig(save_dir / files["stage_comparison"], dpi=160)
    comparison_fig.savefig(save_dir / files["comparison"], dpi=160)

    threshold = float(initial_scores.mean())
    query_nodes = np.flatnonzero(initial_scores >= threshold)
    H, W = labels.shape
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
            "sigma_square": args.sigma_square,
            "alpha": args.alpha,
        },
        "result": {
            "n_segments": n_nodes,
            "labels_shape": list(labels.shape),
            "saliency_shape": list(final_image.shape),
            "area_min": int(sizes.min()),
            "area_max": int(sizes.max()),
            "area_mean": float(sizes.mean()),
            "area_total": int(sizes.sum()),
            "foreground_threshold": threshold,
            "foreground_query_nodes": query_nodes.tolist(),
            "foreground_query_count": len(query_nodes),
            "foreground_query_pixels": int(query_image.sum()),
            "saliency_min": float(final_image.min()),
            "saliency_max": float(final_image.max()),
            "saliency_mean": float(final_image.mean()),
        },
        "timing_seconds": {
            "detection": detection_seconds,
            "visualization": visualization_seconds,
        },
        "files": files,
    }
    with (save_dir / files["metadata"]).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"图名：{image_name}")
    print(f"目标超像素数：{args.n_segments}，实际超像素数：{n_nodes}")
    print(f"前景查询节点数：{len(query_nodes)}，阈值：{threshold:.6f}")
    print(f"检测耗时：{detection_seconds:.3f} 秒")
    print(f"保存目录：{save_dir.resolve()}")

    if args.show:
        plt.show()
    plt.close(stage_fig)
    plt.close(comparison_fig)


if __name__ == "__main__":
    main()
