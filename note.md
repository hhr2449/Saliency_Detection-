# 基于图形流排序的显著性检测

## SLIC

```mermaid
flowchart TD
    A["输入图像<br/>期望区域数 K、紧致度 m"] --> B["转换到 Lab 颜色空间"]
    B --> C["均匀放置初始中心<br/>在小邻域内调整位置"]
    C --> D["像素分配<br/>比较附近中心，选择综合距离最小者"]
    D --> E["中心更新<br/>计算每组像素的平均颜色和平均位置"]
    E --> F{"是否满足停止条件？"}
    F -- 否 --> D
    F -- 是 --> G["连通性处理<br/>整理孤立碎片"]
    G --> H["输出超像素标签图"]
```

这一步的核心就是将整个图像分成一个一个的小区域，这些小区域会被组织成超像素。**后续图流形排序以这些超像素为节点计算显著程度。** 相比于按像素进行显著度计算，这样可以大大减少后续图排序的计算量

1. 首先我们将图像转换到Lab颜色空间，p=[L,a,b,x,y]，其中前三项描述颜色，后两项描述位置。聚类的中心同样使用这样的P来表示

2. 接着我们计算距离。从直观上来想，距离应该包含两个层面：空间上的距离和颜色上的距离。颜色越近，距离越近越应该认为属于一个区域。所以分别计算这两个维度的欧氏距离，然后再取加权欧氏距离，具体而言：

   先计算颜色距离：

   $ d_c= \sqrt{(L-L_k)^2+(a-a_k)^2+(b-b_k)^2} $

   再计算空间距离：

   $d_s= \sqrt{(x-x_k)^2+(y-y_k)^2} $

   最后合成：

   $D=\sqrt{d_c^2+\left(\frac{m}{S}\right)^2d_s^2}$

   这里的系数:

   S 是初始中心之间的大致间隔。

   m 是紧致度参数，控制空间距离的重要程度。m大，则更重视空间的接近性

3. 然后我们在图上放置一些初始的区域中心，根据我们期望的区域数来均匀放置

   假设图像有 $H\times W$个像素，希望得到约 \(K\) 个超像素，那么每个区域的平均面积约为：

   $\frac{HW}{K} $

   把它近似看成正方形，边长就是：

   $S\approx\sqrt{\frac{HW}{K}} $

   将整个图近似初始化为一个个的正方形区域

   > [!tip]
   >
   > 要避开边缘位置
   >
   > 先均匀放置中心，再在每个中心周围的 \(3\times3\) 邻域内，找到梯度最小的位置，把中心移动过去

4. 然后开始循环更新区域和像素中心：

   1. 取中心附近2S*2S的区域
   2. 计算像素与中心的综合距离，一个像素可能归属到多个不同中心的区域中，然后将其归入到距离最小的中心
   3. 在区域范围确定下来之后再在区域中取平均值，作为新的区域中心:$ C_k^{\text{new}} = [\overline L_k,\overline a_k,\overline b_k,\overline x_k,\overline y_k] $

5. 最后对这些区域调整连通性

   上一步结束后会输出一个标签图。这个标签图和原图大小相等，对应的就是原图中每个像素所属的区域标签。连通信调整就是要把同一个标签的像素连成一块。这里采用的是将小的连通块合并到周边区域的方法

   比如：

   ```
   A A A B B B B
   A A A B B A B
   A A A B B B B
   C C C C B B B
   C C C C B B B
   ```

   调整后就是：

   ```
   A A A B B B B
   A A A B B B B
   A A A B B B B
   C C C C B B B
   C C C C B B B
   ```

   

最终输出一个标签图提供给下一步使用





| 顺序 | 本次只完成的内容                                             | 完成标准                                                     | 建议提交信息                                           |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------ |
| 0    | 保存现有目录与代码骨架                                       | 文件已纳入版本管理，包能导入                                 | `chore: add SLIC project skeleton`                     |
| 1    | 添加依赖声明和 `.gitignore`                                  | 声明 NumPy、Pillow、scikit-image、pytest；忽略缓存和生成结果 | `chore: configure dependencies and ignored files`      |
| 2    | `image.prepare_rgb()`                                        | 正确转换 uint8、浮点输入；拒绝非法尺寸、范围和非有限值       | `feat: validate and normalize RGB input`               |
| 3    | `image.load_rgb()`                                           | 能读取图片，并返回统一格式的 RGB 数组                        | `feat: load images as RGB arrays`                      |
| 4    | `image.rgb_to_lab()`                                         | 转换后尺寸正确，已知颜色的 Lab 值合理                        | `feat: convert RGB images to Lab`                      |
| 5    | `initialization.compute_gradient()`                          | 纯色图梯度为零；颜色交界处有响应；边缘不越界                 | `feat: compute Lab image gradients`                    |
| 6    | `initialization.initialize_centers()`                        | 网格中心分布合理，坐标有效，返回参考间隔 `S`                 | `feat: initialize SLIC centers on a grid`              |
| 7    | `initialization.perturb_centers()`                           | 中心移动到局部低梯度位置，同时更新颜色和坐标                 | `feat: move initial centers to low-gradient positions` |
| 8    | `visualization.find_boundaries()`                            | 人工标签图的水平、竖直边界提取正确                           | `feat: extract superpixel boundaries`                  |
| 9    | `visualization.overlay_boundaries()`                         | 能叠加边界，并保持输入图像不被修改                           | `feat: overlay superpixel boundaries on images`        |
| 10   | 在 `clustering.py` 新增私有距离函数 `_compute_distance_squared()` | 颜色项、空间项及权重能通过手算样例核对                       | `feat: compute SLIC color-spatial distances`           |
| 11   | `clustering.assign_pixels()`                                 | 局部窗口内竞争分配正确，每轮重置距离，最终无未分配像素       | `feat: assign pixels to nearby SLIC centers`           |
| 12   | `clustering.compute_region_stats()`                          | 人工区域的面积、平均颜色和平均位置计算正确                   | `feat: compute superpixel region statistics`           |
| 13   | `clustering.update_centers()`                                | 正确更新非空中心，空簇保留旧中心，无除零问题                 | `feat: update SLIC cluster centers`                    |
| 14   | `connectivity.split_components()`                            | 同一标签的不同四连通块获得不同编号                           | `feat: split labels into connected components`         |
| 15   | `connectivity.relabel_contiguous()`                          | 任意有效标签被映射成连续编号，区域归属不变                   | `feat: relabel regions with contiguous identifiers`    |
| 16   | `connectivity.merge_small_components()`                      | 小块并入共享边界最长的邻区；连续多次合并仍正确               | `feat: merge small adjacent components`                |
| 17   | `connectivity.enforce_connectivity()`                        | 串联拆分、合并、编号，每个最终标签对应一个连通区域           | `feat: enforce superpixel connectivity`                |
| 18   | `core.segment_slic()`                                        | 串联完整算法，检查参数，并重新统计最终区域特征               | `feat: assemble the complete SLIC pipeline`            |
| 19   | `demo_slic.main()` 的读取与运行部分                          | 能从命令行读取真实图片，运行分割，输出实际区域数             | `feat: run SLIC from the command line`                 |
| 20   | 演示程序的结果保存功能                                       | 保存标签、边界图及参数记录                                   | `feat: save SLIC segmentation results`                 |
| 21   | 完整流程的集成验证                                           | 纯色、双色、小图、长条图和 `K=1` 均满足最终输出约定          | `test: verify the complete SLIC pipeline`              |
| 22   | README 使用说明                                              | 写明安装、运行命令、输出文件和参数含义                       | `docs: document SLIC usage and parameters`             |