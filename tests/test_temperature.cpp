#include "cuda_utils.hpp"
#include "simulation.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// 与 GPU 代码一致的行优先索引，用于独立的 CPU 参考计算。
std::size_t index(int i, int j, int nx) {
    return static_cast<std::size_t>(j) * nx + i;
}

void apply_cpu_boundaries(std::vector<double>& temperature, const geodynamics::Grid2D& grid,
                          const geodynamics::ThermalParameters& parameters) {
    // 上下 Dirichlet 固定温度边界，先赋值以确定四个角的温度。
    for (int i = 0; i < grid.nx; ++i) {
        temperature[index(i, 0, grid.nx)] = parameters.bottom_temperature;
        temperature[index(i, grid.ny - 1, grid.nx)] = parameters.top_temperature;
    }
    // 左右 Neumann 零梯度边界：仅处理非角点，复制相邻内部温度。
    for (int j = 1; j < grid.ny - 1; ++j) {
        temperature[index(0, j, grid.nx)] = temperature[index(1, j, grid.nx)];
        temperature[index(grid.nx - 1, j, grid.nx)] = temperature[index(grid.nx - 2, j, grid.nx)];
    }
}

std::vector<double> cpu_diffusion_step(const std::vector<double>& temperature,
                                       const geodynamics::Grid2D& grid,
                                       const geodynamics::ThermalParameters& parameters) {
    // CPU 版本有意采用与 GPU kernel 相同的公式，作为逐元素结果基准。
    std::vector<double> next = temperature;
    for (int j = 1; j < grid.ny - 1; ++j) {
        for (int i = 1; i < grid.nx - 1; ++i) {
            const std::size_t center_index = index(i, j, grid.nx);
            // 五点中心差分拉普拉斯算子，边界单元不进入该循环。
            const double center = temperature[center_index];
            const double laplacian_x = (temperature[index(i - 1, j, grid.nx)] - 2.0 * center +
                                        temperature[index(i + 1, j, grid.nx)]) /
                                       (grid.dx * grid.dx);
            const double laplacian_y = (temperature[index(i, j - 1, grid.nx)] - 2.0 * center +
                                        temperature[index(i, j + 1, grid.nx)]) /
                                       (grid.dy * grid.dy);
            next[center_index] = center + parameters.dt * parameters.diffusivity *
                                               (laplacian_x + laplacian_y);
        }
    }
    return next;
}

// CPU 参考版的解析旋涡速度；必须与 CUDA 版本使用相同流函数。
void vortex_velocity(int i, int j, const geodynamics::Grid2D& grid,
                     const geodynamics::VortexParameters& vortex, double& u, double& v) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double length_x = static_cast<double>(grid.nx - 1) * grid.dx;
    const double length_y = static_cast<double>(grid.ny - 1) * grid.dy;
    const double x = i * grid.dx;
    const double y = j * grid.dy;
    const double sx = std::sin(pi * x / length_x);
    const double sy = std::sin(pi * y / length_y);
    u = vortex.strength * (pi / length_y) * sx * sx * std::sin(2.0 * pi * y / length_y);
    v = -vortex.strength * (pi / length_x) * std::sin(2.0 * pi * x / length_x) * sy * sy;
}

std::vector<double> cpu_advection_diffusion_step(const std::vector<double>& temperature,
                                                 const geodynamics::Grid2D& grid,
                                                 const geodynamics::ThermalParameters& thermal,
                                                 const geodynamics::VortexParameters& vortex) {
    std::vector<double> next = temperature;
    for (int j = 1; j < grid.ny - 1; ++j) {
        for (int i = 1; i < grid.nx - 1; ++i) {
            const std::size_t center_index = index(i, j, grid.nx);
            const double center = temperature[center_index];
            const double left = temperature[index(i - 1, j, grid.nx)];
            const double right = temperature[index(i + 1, j, grid.nx)];
            const double below = temperature[index(i, j - 1, grid.nx)];
            const double above = temperature[index(i, j + 1, grid.nx)];
            double u = 0.0;
            double v = 0.0;
            vortex_velocity(i, j, grid, vortex, u, v);
            const double dtdx = u >= 0.0 ? (center - left) / grid.dx : (right - center) / grid.dx;
            const double dtdy = v >= 0.0 ? (center - below) / grid.dy : (above - center) / grid.dy;
            const double laplacian = (left - 2.0 * center + right) / (grid.dx * grid.dx) +
                                     (below - 2.0 * center + above) / (grid.dy * grid.dy);
            next[center_index] = center + thermal.dt *
                (-u * dtdx - v * dtdy + thermal.diffusivity * laplacian);
        }
    }
    return next;
}

void require_close(const std::vector<double>& gpu_result, const std::vector<double>& reference,
                   const char* label) {
    double maximum_error = 0.0;
    for (std::size_t cell = 0; cell < gpu_result.size(); ++cell) {
        maximum_error = std::max(maximum_error, std::abs(gpu_result[cell] - reference[cell]));
    }
    if (maximum_error > 1.0e-12) {
        throw std::runtime_error(std::string(label) + " CPU/GPU 最大误差为 " +
                                 std::to_string(maximum_error));
    }
}

}  // namespace

int main() {
    try {
        // 小网格足以覆盖内部点、四条边和四个角，便于定位边界实现问题。
        const geodynamics::Grid2D grid(9, 7, 0.25, 0.25);
        const geodynamics::ThermalParameters parameters{0.5, 0.01, 1.0, 0.0};

        // 构造非均匀初始场，防止常数场掩盖索引或差分错误。
        std::vector<double> input(grid.cell_count());
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                input[index(i, j, grid.nx)] = 0.1 * i + 0.03 * j * j;
            }
        }
        apply_cpu_boundaries(input, grid, parameters);
        std::vector<double> reference = cpu_diffusion_step(input, grid, parameters);
        apply_cpu_boundaries(reference, grid, parameters);

        // GPU 从同一份边界处理后的输入开始，保证比较仅针对实现差异。
        geodynamics::DeviceBuffer<double> temperature(grid.cell_count());
        geodynamics::DeviceBuffer<double> next_temperature(grid.cell_count());
        temperature.copy_from_host(input.data(), grid.cell_count());
        geodynamics::apply_temperature_boundaries(temperature.data(), grid, parameters);
        geodynamics::advance_temperature_diffusion(temperature.data(), next_temperature.data(), grid, parameters);
        geodynamics::apply_temperature_boundaries(next_temperature.data(), grid, parameters);

        std::vector<double> gpu_result(grid.cell_count());
        next_temperature.copy_to_host(gpu_result.data(), grid.cell_count());

        // 使用最大逐元素误差，而不是视觉比较，作为严格验收条件。
        require_close(gpu_result, reference, "纯扩散");

        // 同一初始场再验证一阶迎风平流-扩散，确保速度符号和上游选取正确。
        const geodynamics::VortexParameters vortex{0.03};
        std::vector<double> advection_reference = cpu_advection_diffusion_step(input, grid, parameters, vortex);
        apply_cpu_boundaries(advection_reference, grid, parameters);
        temperature.copy_from_host(input.data(), grid.cell_count());
        geodynamics::advance_temperature_advection_diffusion(
            temperature.data(), next_temperature.data(), grid, parameters, vortex);
        geodynamics::apply_temperature_boundaries(next_temperature.data(), grid, parameters);
        next_temperature.copy_to_host(gpu_result.data(), grid.cell_count());
        require_close(gpu_result, advection_reference, "平流-扩散");

        const auto diagnostics = geodynamics::compute_temperature_diagnostics(
            next_temperature.data(), grid, parameters, vortex);
        if (!diagnostics.is_finite || diagnostics.advection_cfl <= 0.0 ||
            diagnostics.diffusion_cfl <= 0.0 || diagnostics.mean_temperature <= 0.0 ||
            diagnostics.mean_squared_temperature <= 0.0) {
            throw std::runtime_error("温度诊断结果无效");
        }

        std::cout << "温度扩散与平流-扩散 CPU/GPU 对照测试通过\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Temperature diffusion test failed: " << error.what() << '\n';
        return 1;
    }
}
