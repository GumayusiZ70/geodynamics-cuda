#include "cuda_utils.hpp"
#include "simulation.hpp"

#include <stdexcept>

namespace geodynamics {
namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

// 由流函数 psi = strength * sin^2(pi*x/Lx) * sin^2(pi*y/Ly) 导出的封闭旋涡。
// u = d psi / d y，v = -d psi / d x，因此连续意义下自动满足 div(u) = 0。
__device__ void vortex_velocity(int i, int j, int nx, int ny, double dx, double dy,
                                double strength, double& u, double& v) {
    const double length_x = static_cast<double>(nx - 1) * dx;
    const double length_y = static_cast<double>(ny - 1) * dy;
    const double x = static_cast<double>(i) * dx;
    const double y = static_cast<double>(j) * dy;
    const double sx = sin(pi * x / length_x);
    const double sy = sin(pi * y / length_y);

    u = strength * (pi / length_y) * sx * sx * sin(2.0 * pi * y / length_y);
    v = -strength * (pi / length_x) * sin(2.0 * pi * x / length_x) * sy * sy;
}

// 一个线程完成一个温度单元的一步显式扩散：
// T^(n+1) = T^n + dt * kappa * (d2T/dx2 + d2T/dy2)。
__global__ void temperature_diffusion_kernel(const double* temperature, double* next_temperature,
                                             int nx, int ny, double dx, double dy,
                                             double diffusivity, double dt) {
    // 将二维 CUDA 线程坐标映射到温度网格坐标。
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) {
        return;
    }

    const int index = j * nx + i;  // 行优先的一维索引。
    if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        // 边界值由 boundary.cu 单独施加；此 kernel 不改变它们。
        next_temperature[index] = temperature[index];
        return;
    }

    // 内部单元使用五点中心差分计算拉普拉斯算子。
    const double center = temperature[index];
    const double laplacian_x = (temperature[index - 1] - 2.0 * center + temperature[index + 1]) / (dx * dx);
    const double laplacian_y = (temperature[index - nx] - 2.0 * center + temperature[index + nx]) / (dy * dy);
    next_temperature[index] = center + dt * diffusivity * (laplacian_x + laplacian_y);
}

// 一个线程完成一个单元的一阶迎风平流加五点差分扩散。
// 平流项写成 -u*dT/dx-v*dT/dy；差分方向由局部速度符号决定。
__global__ void temperature_advection_diffusion_kernel(const double* temperature,
                                                       double* next_temperature, int nx, int ny,
                                                       double dx, double dy, double diffusivity,
                                                       double dt, double vortex_strength) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) {
        return;
    }

    const int index = j * nx + i;
    if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        next_temperature[index] = temperature[index];
        return;
    }

    double u = 0.0;
    double v = 0.0;
    vortex_velocity(i, j, nx, ny, dx, dy, vortex_strength, u, v);

    const double center = temperature[index];
    const double left = temperature[index - 1];
    const double right = temperature[index + 1];
    const double below = temperature[index - nx];
    const double above = temperature[index + nx];

    // 迎风差分只读取流动上游的温度，牺牲部分精度以换取初版的稳定性。
    const double dtdx = u >= 0.0 ? (center - left) / dx : (right - center) / dx;
    const double dtdy = v >= 0.0 ? (center - below) / dy : (above - center) / dy;
    const double laplacian = (left - 2.0 * center + right) / (dx * dx) +
                             (below - 2.0 * center + above) / (dy * dy);
    next_temperature[index] = center + dt * (-u * dtdx - v * dtdy + diffusivity * laplacian);
}

}  // namespace

void advance_temperature_diffusion(const double* temperature, double* next_temperature,
                                   const Grid2D& grid, const ThermalParameters& parameters) {
    // 当前显式格式仅检查正值；完整 CFL/扩散稳定条件将在 diagnostics 模块中加入。
    if (parameters.diffusivity <= 0.0 || parameters.dt <= 0.0) {
        throw std::invalid_argument("Thermal diffusivity and time step must be positive");
    }

    // 向上取整，确保边缘线程块也覆盖所有网格点；越界线程在 kernel 开头返回。
    constexpr int block_size = 16;
    const dim3 block(block_size, block_size);
    const dim3 launch_grid((grid.nx + block.x - 1) / block.x,
                           (grid.ny + block.y - 1) / block.y);
    temperature_diffusion_kernel<<<launch_grid, block>>>(
        temperature, next_temperature, grid.nx, grid.ny, grid.dx, grid.dy,
        parameters.diffusivity, parameters.dt);
    CUDA_CHECK(cudaGetLastError());
}

void advance_temperature_advection_diffusion(const double* temperature, double* next_temperature,
                                             const Grid2D& grid, const ThermalParameters& thermal,
                                             const VortexParameters& vortex) {
    if (thermal.diffusivity <= 0.0 || thermal.dt <= 0.0) {
        throw std::invalid_argument("热扩散系数和时间步长必须为正值");
    }

    constexpr int block_size = 16;
    const dim3 block(block_size, block_size);
    const dim3 launch_grid((grid.nx + block.x - 1) / block.x,
                           (grid.ny + block.y - 1) / block.y);
    temperature_advection_diffusion_kernel<<<launch_grid, block>>>(
        temperature, next_temperature, grid.nx, grid.ny, grid.dx, grid.dy,
        thermal.diffusivity, thermal.dt, vortex.strength);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace geodynamics
