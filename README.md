# 基于图流形排序的显著区域检测

媒体计算课程大作业，复现 *Saliency Detection via Graph-Based Manifold Ranking*（CVPR 2013）。输入一幅图像，输出与原图大小相同的像素级显著图

核心实现包括自行实现的 SLIC 超像素分割、超像素图构建、非归一化流形排序，以及背景查询和前景查询两个阶段。显著图中，白色表示显著程度高，黑色表示显著程度低

项目结构：

```
Saliency_Detection/
├── src/
│   ├── slic/              # 自行实现的 SLIC
│   ├── graph/             # 超像素邻接关系和颜色边权
│   ├── ranking/           # 关联矩阵与查询分数计算
│   ├── saliency/          # 背景查询、前景查询、归一化
│   ├── visualization/     # SLIC 与显著性结果展示
│   └── pipeline.py        # 全流程检测入口
├── demo/
│   ├── demo_slic.py
│   └── demo_saliency.py
├── tests/                 # 模块测试及流程衔接测试
├── images/                # 输入图片
├── outputs/               # Demo 输出
├── requirements.txt
├── AI_LOG.md              # AI 使用记录
└── README.md
```



## 1. 算法流程

1. 将 RGB 图像转换到 Lab 颜色空间，使用 SLIC 划分超像素，并调整区域连通性
2. 以超像素为节点，连接空间相邻区域、邻居的邻居，以及图像四侧的边界节点。根据区域平均 Lab 颜色计算边权
3. 计算度矩阵和关联矩阵：

   $$D_{ii}=\sum_j W_{ij},\qquad B=(D-\alpha W)^{-1}$$

   将关联矩阵的对角线清零，去掉节点对自身的关联
4. 分别以四侧边界节点为背景查询，计算关联分数；每侧归一化并取反，然后逐元素相乘、归一化，得到第一阶段显著分数
5. 将不低于第一阶段超像素平均显著值的节点选为前景查询，复用关联矩阵进行排序，并归一化得到最终分数
6. 通过标签图将超像素分数映射回像素，生成显著图

## 2. 环境与依赖

使用 Python 3.10 或以上版本。以下命令均在项目根目录执行

```bash
python -m pip install -r requirements.txt
```

| 依赖 | 用途 |
|---|---|
| NumPy | 数组计算、区域统计、矩阵运算和求逆 |
| Pillow | 图像读取、RGB 转换及 PNG 保存 |
| scikit-image | RGB 到 Lab 的颜色空间转换 |
| Matplotlib | 绘制边界图、显著图和对比图 |
| pytest | 测试代码使用的框架 |

## 3. 快速运行

### 完整显著性检测

```bash
python demo/demo_saliency.py images/horse.jpg
```

默认保存到项目根目录下的 `outputs/saliency/horse/`

指定图名并显示对比图：

```bash
python demo/demo_saliency.py images/horse.jpg --name horse_demo --show
```

自定义保存位置和算法参数：

```bash
python demo/demo_saliency.py images/horse.jpg --save-dir outputs/saliency/horse_k300 --n-segments 300 --compactness 10 --alpha 0.99 --sigma-square 0.1
```

### SLIC 分割展示

```bash
python demo/demo_slic.py images/horse.jpg
```

默认保存到项目根目录下的 `outputs/slic/horse/`

```bash
python demo/demo_slic.py images/horse.jpg --name horse_demo --n-segments 200 --compactness 10 --show
```

查看完整命令行帮助：

```bash
python demo/demo_saliency.py --help
python demo/demo_slic.py --help
```

## 4. 命令行参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `image` | 必填 | 输入图像路径 |
| `--name` / `--image-name` | 源文件名，不含扩展名 | 输出图名，例如 `horse_demo` |
| `--save-dir` / `--output-dir` | 对应 Demo 的默认结果目录 | 直接指定完整输出目录，不再额外拼接图名 |
| `--n-segments` | `200` | 期望超像素数量 K；实际数量会受到网格初始化和连通性处理影响 |
| `--compactness` | `10.0` | SLIC 紧致度；越大越重视空间距离 |
| `--max-iter` | `10` | SLIC 聚类迭代次数 |
| `--min-size-factor` | `0.25` | 小连通块面积阈值相对于初始平均区域面积的比例 |
| `--sigma-square` | `0.1` | 边权公式的分母 σ²，仅完整检测 Demo 使用 |
| `--alpha` | `0.99` | 流形排序参数，取值满足 0 < α < 1，仅完整检测 Demo 使用 |
| `--show` | 关闭 | 保存结果后显示对比图窗口 |

## 5. 输出文件

### 显著性检测

| 文件 | 内容 |
|---|---|
| `original.png` | 输入图像转换为 RGB 后的原图 |
| `initial_saliency.png` | 四侧背景查询融合后的第一阶段显著图 |
| `foreground_queries.png` | 前景查询二值图，白色区域表示入选节点 |
| `final_saliency.png` | 最终像素级显著图 |
| `stage_comparison.png` | 第一阶段与最终显著图的并排对比 |
| `comparison.png` | 原图、第一阶段显著图、前景查询二值图、最终显著图的四联图 |
| `metadata.json` | 输入信息、参数、区域统计、前景查询统计、显著度统计、耗时和文件名 |

### SLIC

| 文件 | 内容 |
|---|---|
| `original.png` | RGB 原图 |
| `boundaries.png` | 在原图上叠加超像素边界 |
| `mean_colors.png` | 用区域平均 RGB 颜色替换区域内像素 |
| `comparison.png` | 原图、区域边界图、区域平均颜色图的三联图 |
| `metadata.json` | 输入信息、SLIC 参数、实际区域数、面积统计、耗时和文件名 |

