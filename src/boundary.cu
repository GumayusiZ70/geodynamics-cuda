#include "cuda_utils.hpp"
#include "simulation.hpp"

namespace geodynamics {
namespace {

// 一个线程处理一个标量单元的边界值。
// 边界优先级为上下固定温度，再处理左右绝热，保证四个角由上下温度决定。
__global__ void apply_temperature_boundaries_kernel(double* temperature, int nx, int ny,
                                                     double bottom_temperature,
                                                     double top_temperature) {
    // 二维线程块坐标映射到网格列 i 和行 j。
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) {
        return;
    }

    const int index = j * nx + i;  // 行优先压平索引。
    if (j == 0) {
        temperature[index] = bottom_temperature;
    } else if (j == ny - 1) {
        temperature[index] = top_temperature;
    } else if (i == 0) {
        // 左右绝热：边界值复制相邻内部单元，使 dT/dx = 0。
        temperature[index] = temperature[j * nx + 1];
    } else if (i == nx - 1) {
        temperature[index] = temperature[j * nx + nx - 2];
    }
}

}  // namespace

void apply_temperature_boundaries(double* temperature, const Grid2D& grid,
                                  const ThermalParameters& parameters) {
    // 16 x 16 线程块适合当前规则二维 stencil；网格尺寸不必是 16 的整数倍。
    constexpr int block_size = 16;
    const dim3 block(block_size, block_size);
    const dim3 launch_grid((grid.nx + block.x - 1) / block.x,
                           (grid.ny + block.y - 1) / block.y);
    apply_temperature_boundaries_kernel<<<launch_grid, block>>>(
        temperature, grid.nx, grid.ny, parameters.bottom_temperature, parameters.top_temperature);
    // kernel 启动是异步的；此处先检查启动参数或资源配置错误。
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace geodynamics
