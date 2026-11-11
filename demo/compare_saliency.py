"""对同一张图片比较不同 K 的显著性结果、实际超像素数量和完整检测耗时"""

from __future__ import annotations

import argparse
import csv
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
    """读取输入图片、K 列表和本次对照中固定的算法参数"""
    parser = argparse.ArgumentParser(description="期望超像素数量 K 的参数对照实验")
    parser.add_argument("image", type=Path, help="输入图像路径")
    parser.add_argument(
        "--name", "--image-name", dest="image_name",
        help="输出图名，默认使用源文件名（不含扩展名）",
    )
    parser.add_argument(
        "--k-values", nargs="+", type=int, default=[50, 100, 200, 400, 800],
        help="依次比较的期望超像素数量，默认 50 100 200 400 800",
    )
    parser.add_argument("--compactness", type=float, default=10.0, help="固定的 SLIC 紧致度参数")
    parser.add_argument("--max-iter", type=int, default=10, help="固定的 SLIC 聚类迭代次数")
    parser.add_argument("--min-size-factor", type=float, default=0.25, help="固定的小区域面积阈值比例")
    parser.add_argument("--sigma-square", type=float, default=0.1, help="固定的颜色边权参数 sigma²")
    parser.add_argument("--alpha", type=float, default=0.99, help="固定的流形排序参数")
    parser.add_argument("--show", action="store_true", help="保存后显示参数对比图")
    return parser


def save_image(image: np.ndarray, image_path: Path) -> None:
    """将 [0, 1] 的 RGB 或灰度图像保存为 uint8 PNG"""
    pixels = np.rint(image * 255).astype(np.uint8)
    Image.fromarray(pixels).save(image_path)


def save_summary(rows: list[dict], save_dir: Path) -> None:
    """将输入图片、期望 K、实际 N 和检测耗时保存为 CSV 与 Markdown 表格"""
    headers = ["输入图片", "期望数量 K", "实际超像素数量 N", "完整检测耗时（秒）"]

    with (save_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([
                row["image"], row["n_segments"], row["actual_segments"],
                f'{row["detection_seconds"]:.6f}',
            ])

    lines = [
        "| " + " | ".join(headers) + " |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f'| {row["image"]} | {row["n_segments"]} | '
            f'{row["actual_segments"]} | {row["detection_seconds"]:.6f} |'
        )
    (save_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    image_name = args.image_name or args.image.stem
    save_dir = PROJECT_ROOT / "outputs" / "compare" / image_name

    # 先设置绘图后端，再导入可视化接口
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from visualization import plot_k_comparison

    image = load_rgb(args.image)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_image(image, save_dir / "original.png")

    fixed_parameters = {
        "compactness": args.compactness,
        "max_iter": args.max_iter,
        "min_size_factor": args.min_size_factor,
        "sigma_square": args.sigma_square,
        "alpha": args.alpha,
    }
    files = {
        "original": "original.png",
        "comparison": "comparison.png",
        "summary_csv": "summary.csv",
        "summary_markdown": "summary.md",
        "metadata": "metadata.json",
    }
    rows = []
    saliency_maps = []
    actual_counts = []

    print("输入图片\tK\tN\t完整检测耗时（秒）", flush=True)
    for k in args.k_values:
        # 只计时完整算法调用，其余参数在所有 K 下保持一致
        start = perf_counter()
        result = detect_saliency(
            image, n_segments=k, return_details=True, **fixed_parameters,
        )
        detection_seconds = perf_counter() - start

        n_actual = len(result["sizes"])
        saliency_map = result["saliency_map"]
        filename = f"k_{k}.png"
        save_image(saliency_map, save_dir / filename)
        files[f"k_{k}"] = filename
        saliency_maps.append(saliency_map)
        actual_counts.append(n_actual)
        rows.append({
            "image": args.image.name,
            "n_segments": k,
            "actual_segments": n_actual,
            "detection_seconds": detection_seconds,
            "saliency_file": filename,
        })
        print(f"{args.image.name}\t{k}\t{n_actual}\t{detection_seconds:.6f}", flush=True)

    # 使用同一灰度范围并排展示所有 K 对应的最终显著图
    fig = plot_k_comparison(
        image, saliency_maps, args.k_values, actual_counts,
        title=f"{image_name} | 期望超像素数量 K 对照",
    )
    fig.savefig(save_dir / files["comparison"], dpi=160)
    save_summary(rows, save_dir)

    H, W = image.shape[:2]
    metadata = {
        "image": {
            "name": image_name,
            "source_path": str(args.image.resolve()),
            "height": H,
            "width": W,
            "channels": image.shape[2],
        },
        "comparison": {
            "parameter": "n_segments",
            "values": args.k_values,
        },
        "fixed_parameters": fixed_parameters,
        "timing_scope": "仅统计 detect_saliency 调用，不含图像读取、绘图和文件保存",
        "runs": rows,
        "files": files,
    }
    with (save_dir / files["metadata"]).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"保存目录：{save_dir.resolve()}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
