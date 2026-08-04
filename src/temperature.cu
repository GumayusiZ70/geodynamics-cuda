#include "cuda_utils.hpp"
#include "simulation.hpp"

#include <stdexcept>

namespace geodynamics {
namespace {

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

}  // namespace geodynamics
