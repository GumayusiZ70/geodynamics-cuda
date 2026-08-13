#include "cuda_utils.hpp"
#include "simulation.hpp"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <cmath>
#include <limits>

namespace geodynamics {
namespace {

// Thrust 归约使用的逐元素变换：先取绝对值，再由 maximum 求最大值。
struct AbsoluteValue {
    __host__ __device__ double operator()(double value) const {
        return value < 0.0 ? -value : value;
    }
};

// 供 Thrust transform_reduce 使用，计算温度平方平均值这一能量代理量。
struct SquareValue {
    __host__ __device__ double operator()(double value) const {
        return value * value;
    }
};

constexpr double pi = 3.141592653589793238462643383279502884;

__global__ void vortex_speed_kernel(double* abs_u, double* abs_v, int nx, int ny, double dx,
                                    double dy, double strength) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) {
        return;
    }

    const int index = j * nx + i;
    const double length_x = static_cast<double>(nx - 1) * dx;
    const double length_y = static_cast<double>(ny - 1) * dy;
    const double x = static_cast<double>(i) * dx;
    const double y = static_cast<double>(j) * dy;
    const double sx = sin(pi * x / length_x);
    const double sy = sin(pi * y / length_y);
    const double u = strength * (pi / length_y) * sx * sx * sin(2.0 * pi * y / length_y);
    const double v = -strength * (pi / length_x) * sin(2.0 * pi * x / length_x) * sy * sy;
    abs_u[index] = u < 0.0 ? -u : u;
    abs_v[index] = v < 0.0 ? -v : v;
}

__global__ void finite_flag_kernel(const double* values, std::size_t count, int* invalid_flag) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && !isfinite(values[index])) {
        atomicExch(invalid_flag, 1);
    }
}

}  // namespace

double max_abs_device(const double* values, std::size_t count) {
    // 归约全过程留在 GPU，避免为一个诊断量复制整块数组回 CPU。
    thrust::device_ptr<const double> begin(values);
    return thrust::transform_reduce(thrust::device, begin, begin + count, AbsoluteValue{}, 0.0,
                                    thrust::maximum<double>());
}

TemperatureDiagnostics compute_temperature_diagnostics(const double* temperature, const Grid2D& grid,
                                                       const ThermalParameters& thermal,
                                                       const VortexParameters& vortex) {
    const std::size_t count = grid.cell_count();
    thrust::device_ptr<const double> begin(temperature);
    const double minimum = thrust::reduce(thrust::device, begin, begin + count,
                                          std::numeric_limits<double>::infinity(), thrust::minimum<double>());
    const double maximum = thrust::reduce(thrust::device, begin, begin + count,
                                          -std::numeric_limits<double>::infinity(), thrust::maximum<double>());
    const double sum = thrust::reduce(thrust::device, begin, begin + count, 0.0, thrust::plus<double>());
    const double sum_of_squares = thrust::transform_reduce(
        thrust::device, begin, begin + count, SquareValue{}, 0.0, thrust::plus<double>());

    DeviceBuffer<double> abs_u(count);
    DeviceBuffer<double> abs_v(count);
    constexpr int block_size_2d = 16;
    const dim3 block_2d(block_size_2d, block_size_2d);
    const dim3 grid_2d((grid.nx + block_2d.x - 1) / block_2d.x,
                       (grid.ny + block_2d.y - 1) / block_2d.y);
    vortex_speed_kernel<<<grid_2d, block_2d>>>(abs_u.data(), abs_v.data(), grid.nx, grid.ny,
                                                grid.dx, grid.dy, vortex.strength);
    CUDA_CHECK(cudaGetLastError());
    const double max_u = max_abs_device(abs_u.data(), count);
    const double max_v = max_abs_device(abs_v.data(), count);

    DeviceBuffer<int> invalid_flag(1);
    CUDA_CHECK(cudaMemset(invalid_flag.data(), 0, sizeof(int)));
    constexpr int block_size_1d = 256;
    const int blocks = static_cast<int>((count + block_size_1d - 1) / block_size_1d);
    finite_flag_kernel<<<blocks, block_size_1d>>>(temperature, count, invalid_flag.data());
    CUDA_CHECK(cudaGetLastError());
    int host_invalid_flag = 0;
    invalid_flag.copy_to_host(&host_invalid_flag, 1);

    return TemperatureDiagnostics{
        minimum,
        maximum,
        sum / static_cast<double>(count),
        sum_of_squares / static_cast<double>(count),
        thermal.dt * (max_u / grid.dx + max_v / grid.dy),
        thermal.diffusivity * thermal.dt * (1.0 / (grid.dx * grid.dx) + 1.0 / (grid.dy * grid.dy)),
        host_invalid_flag == 0,
    };
}

}  // namespace geodynamics
