#include "cuda_utils.hpp"
#include "stokes.hpp"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <stdexcept>

namespace geodynamics {
namespace {

// 每个线程处理一个单元中心，使用相邻四个速度面计算离散散度。
__global__ void mac_divergence_kernel(const double* u, const double* v, double* divergence,
                                      int nx, int ny, double dx, double dy) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) {
        return;
    }

    const int scalar = j * nx + i;
    const int u_left = j * (nx + 1) + i;
    const int u_right = u_left + 1;
    const int v_bottom = j * nx + i;
    const int v_top = v_bottom + nx;
    divergence[scalar] = (u[u_right] - u[u_left]) / dx + (v[v_top] - v[v_bottom]) / dy;
}

// 每个线程处理一个 u 面。内部面两侧各有一个压力单元，边界面暂置零。
__global__ void pressure_gradient_u_kernel(const double* pressure, double* gradient_u,
                                           int nx, int ny, double dx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > nx || j >= ny) {
        return;
    }

    const int face = j * (nx + 1) + i;
    gradient_u[face] = (i == 0 || i == nx) ? 0.0 :
        (pressure[j * nx + i] - pressure[j * nx + i - 1]) / dx;
}

// 每个线程处理一个 v 面。内部面上下各有一个压力单元，边界面暂置零。
__global__ void pressure_gradient_v_kernel(const double* pressure, double* gradient_v,
                                           int nx, int ny, double dy) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j > ny) {
        return;
    }

    const int face = j * nx + i;
    gradient_v[face] = (j == 0 || j == ny) ? 0.0 :
        (pressure[j * nx + i] - pressure[(j - 1) * nx + i]) / dy;
}

// 一个线程更新一个 u 面的一步伪瞬态动量：u += lambda_u * (eta*lap(u) - dp/dx + fx)。
// 自由滑移：左右壁 (i=0,nx) 法向速度置零；上下壁切向速度用镜像 ghost 使 du/dy=0。
__global__ void momentum_u_kernel(const double* u, const double* pressure, const double* fx,
                                  double* u_next, int nx, int ny, double dx, double dy,
                                  double viscosity, double relaxation) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > nx || j >= ny) {
        return;
    }

    const int face = j * (nx + 1) + i;
    if (i == 0 || i == nx) {
        u_next[face] = 0.0;  // 左右壁法向速度为零。
        return;
    }

    const double center = u[face];
    // x 向中心差分；i=1 与 i=nx-1 处直接使用边界 Dirichlet 值 u(0)=u(nx)=0。
    const double lap_x = (u[face - 1] - 2.0 * center + u[face + 1]) / (dx * dx);
    // y 向中心差分；j=0 与 j=ny-1 处用自由滑移镜像 u(i,-1)=u(i,0)，退化为单侧差分。
    const double below = (j > 0) ? u[face - (nx + 1)] : center;
    const double above = (j < ny - 1) ? u[face + (nx + 1)] : center;
    const double lap_y = (below - 2.0 * center + above) / (dy * dy);

    // u 面压力梯度 (p[i] - p[i-1]) / dx。
    const double grad_p = (pressure[j * nx + i] - pressure[j * nx + i - 1]) / dx;
    u_next[face] = center + relaxation * (viscosity * (lap_x + lap_y) - grad_p + fx[face]);
}

// 一个线程更新一个 v 面的一步伪瞬态动量。
__global__ void momentum_v_kernel(const double* v, const double* pressure, const double* fy,
                                  double* v_next, int nx, int ny, double dx, double dy,
                                  double viscosity, double relaxation) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j > ny) {
        return;
    }

    const int face = j * nx + i;
    if (j == 0 || j == ny) {
        v_next[face] = 0.0;  // 上下壁法向速度为零。
        return;
    }

    const double center = v[face];
    // y 向中心差分；j=1 与 j=ny-1 处直接使用边界 Dirichlet 值 v(0)=v(ny)=0。
    const double lap_y = (v[face - nx] - 2.0 * center + v[face + nx]) / (dy * dy);
    // x 向中心差分；i=0 与 i=nx-1 处用自由滑移镜像 v(-1,j)=v(0,j)。
    const double left = (i > 0) ? v[face - 1] : center;
    const double right = (i < nx - 1) ? v[face + 1] : center;
    const double lap_x = (left - 2.0 * center + right) / (dx * dx);

    // v 面压力梯度 (p[j] - p[j-1]) / dy。
    const double grad_p = (pressure[j * nx + i] - pressure[(j - 1) * nx + i]) / dy;
    v_next[face] = center + relaxation * (viscosity * (lap_x + lap_y) - grad_p + fy[face]);
}

// 一个线程更新一个单元中心的压力：p -= lambda_p * div(u)，散度由刚更新的速度计算。
__global__ void pressure_update_kernel(const double* pressure, const double* u, const double* v,
                                       double* pressure_next, int nx, int ny, double dx, double dy,
                                       double relaxation) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) {
        return;
    }

    const int cell = j * nx + i;
    const double divergence = (u[j * (nx + 1) + i + 1] - u[j * (nx + 1) + i]) / dx +
                             (v[(j + 1) * nx + i] - v[j * nx + i]) / dy;
    pressure_next[cell] = pressure[cell] - relaxation * divergence;
}

// 从每个元素减去标量，用于去除压力均值。
__global__ void subtract_scalar_kernel(double* values, std::size_t count, double scalar) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index] -= scalar;
    }
}

dim3 launch_grid(int width, int height, const dim3& block) {
    return dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
}

}  // namespace

void compute_mac_divergence(const double* u, const double* v, double* divergence,
                            const Grid2D& grid) {
    const dim3 block(16, 16);
    mac_divergence_kernel<<<launch_grid(grid.nx, grid.ny, block), block>>>(
        u, v, divergence, grid.nx, grid.ny, grid.dx, grid.dy);
    CUDA_CHECK(cudaGetLastError());
}

void compute_pressure_gradient(const double* pressure, double* gradient_u, double* gradient_v,
                               const Grid2D& grid) {
    const dim3 block(16, 16);
    pressure_gradient_u_kernel<<<launch_grid(grid.nx + 1, grid.ny, block), block>>>(
        pressure, gradient_u, grid.nx, grid.ny, grid.dx);
    CUDA_CHECK(cudaGetLastError());
    pressure_gradient_v_kernel<<<launch_grid(grid.nx, grid.ny + 1, block), block>>>(
        pressure, gradient_v, grid.nx, grid.ny, grid.dy);
    CUDA_CHECK(cudaGetLastError());
}

void advance_stokes_pseudotransient_step(const double* u, const double* v, const double* pressure,
                                        const double* fx, const double* fy,
                                        double* u_next, double* v_next, double* pressure_next,
                                        const Grid2D& grid, const StokesParameters& parameters) {
    if (parameters.viscosity <= 0.0 || parameters.momentum_relaxation <= 0.0 ||
        parameters.pressure_relaxation <= 0.0) {
        throw std::invalid_argument("Stokes 参数（黏度和两个伪时间步）必须为正值");
    }

    const dim3 block(16, 16);
    momentum_u_kernel<<<launch_grid(grid.nx + 1, grid.ny, block), block>>>(
        u, pressure, fx, u_next, grid.nx, grid.ny, grid.dx, grid.dy,
        parameters.viscosity, parameters.momentum_relaxation);
    CUDA_CHECK(cudaGetLastError());
    momentum_v_kernel<<<launch_grid(grid.nx, grid.ny + 1, block), block>>>(
        v, pressure, fy, v_next, grid.nx, grid.ny, grid.dx, grid.dy,
        parameters.viscosity, parameters.momentum_relaxation);
    CUDA_CHECK(cudaGetLastError());
    // 压力更新读取刚更新的速度（同一默认流上 kernel 顺序执行），保证散度来自新速度。
    pressure_update_kernel<<<launch_grid(grid.nx, grid.ny, block), block>>>(
        pressure, u_next, v_next, pressure_next, grid.nx, grid.ny, grid.dx, grid.dy,
        parameters.pressure_relaxation);
    CUDA_CHECK(cudaGetLastError());
    subtract_mean_pressure(pressure_next, grid);
}

void subtract_mean_pressure(double* pressure, const Grid2D& grid) {
    const std::size_t count = grid.cell_count();
    thrust::device_ptr<double> begin(pressure);
    const double sum = thrust::reduce(thrust::device, begin, begin + count, 0.0,
                                      thrust::plus<double>());
    const double mean = sum / static_cast<double>(count);
    constexpr int block_size = 256;
    const int blocks = static_cast<int>((count + block_size - 1) / block_size);
    subtract_scalar_kernel<<<blocks, block_size>>>(pressure, count, mean);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace geodynamics
