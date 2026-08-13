# geodynamics-cuda

基于 C++/CUDA 的二维不可压缩热地球动力学原型求解器。

## 当前实现范围

当前原型已经实现二维温度场的一阶迎风平流-扩散时间步：底部和顶部为固定温度，左右侧壁为绝热边界。速度由一个解析、无散度的封闭旋涡场给定，因此可以在尚未实现 Stokes 方程前独立验证温度平流。

`Stokes` 伪瞬态和多重网格模块目前是有意保留的占位文件。只有先确定 MAC 网格离散、边界条件和验证算例，才开始实现它们，避免出现“能运行但数值含义不确定”的代码。

## 环境要求

- CMake 3.22 或更高版本
- 支持 C++17 的编译器
- NVIDIA CUDA Toolkit 12 或更高版本

## 构建与测试

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

运行当前的温度扩散演示程序：

```bash
./build/geodynamics
```

## 数组布局

单元中心标量场采用行优先一维存储：

```text
T, p, eta: [ny][nx]
index = j * nx + i
```

后续不可压缩流动使用 MAC 交错网格：

```text
u: [ny][nx + 1]
v: [ny + 1][nx]
```

其中 `u` 位于竖直单元面，`v` 位于水平单元面；`T`、压力 `p` 和黏度 `eta` 位于单元中心。

## 当前输出

运行 `./build/geodynamics` 后，程序会在 `output/` 下每 200 步生成一组文件：

| 文件 | 内容 |
|---|---|
| `temperature_*.csv` | 三列 `x,y,temperature`，适合 Python、ParaView 等工具读取。 |
| `temperature_*.png` | 彩色温度云图，低温为蓝、高温为红，可在 Windows、浏览器和 VS Code 中直接查看。 |
| `temperature_*.ppm` | 与 PNG 内容相同的无压缩图像，用于调试图像写出器。 |
| `diagnostics.csv` | 每个输出步的时间、温度范围、温度平均值、温度平方平均值、CFL 和有限性标志。 |

终端还会输出温度最小/最大值、平均温度、温度平方平均值、平流 CFL、扩散 CFL 和 NaN/Inf 检查结果。温度平方平均值是监测场整体变化的无量纲能量代理量，而不是严格的地球物理热能。当前显式格式要求平流 CFL 保持小于约 1；扩散 CFL 在二维五点差分下应不大于约 0.5。

可选地，安装 `ffmpeg` 后可将连续 PNG 快照制作为动画：

```bash
scripts/make_animation.sh output output/temperature.gif
```

## 文档位置

| 位置 | 用途 |
|---|---|
| `README.md` | 新成员了解项目、配置环境、构建和运行测试的入口。 |
| `docs/文档说明.md` | 文档分类约定，以及后续应补充哪些数值说明。 |
| `docs/roadmap.md` | 按阶段推进的开发路线和验收目标。 |
| `docs/网格与变量布局.md` | MAC 网格中 `T/p/eta/u/v` 的物理位置、数组索引与离散算子。 |
| `docs/边界条件.md` | 温度、速度和压力的边界条件及其离散方式。 |
| `tests/` | CPU 参考计算与 GPU 实现的可执行数值对照。 |

## 协作约定

- `main` 分支必须保持可构建，并通过已有测试。
- 独立功能在 `feature/...` 分支开发，合并前通过 Pull Request 审阅。
- 每个 CUDA 数值 kernel 先有一个小网格 CPU 参考实现，再做并行优化。
- 每个可复现实验记录参数文件、Git commit ID、GPU 型号、CUDA 版本和诊断量。
