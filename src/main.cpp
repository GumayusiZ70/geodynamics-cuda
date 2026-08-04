#include "cuda_utils.hpp"
#include "simulation.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    try {
        // 先确认运行时能识别至少一张 GPU；编译成功并不代表运行时驱动可用。
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            throw std::runtime_error("No CUDA-capable GPU is available");
        }

        // 当前原型固定使用第 0 张卡，后续可从配置文件选择设备编号。
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        CUDA_CHECK(cudaSetDevice(0));

        // 演示算例采用宽高比为 2:1 的二维单元中心温度网格。
        constexpr int nx = 256;
        constexpr int ny = 128;
        const geodynamics::Grid2D grid(nx, ny, 2.0 / (nx - 1), 1.0 / (ny - 1));
        const geodynamics::ThermalParameters thermal{1.0, 2.0e-6, 1.0, 0.0};

        // CPU 端构造“底热顶冷”的线性温度背景，并叠加小扰动触发演示变化。
        std::vector<double> host_temperature(grid.cell_count());
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                const double x = static_cast<double>(i) * grid.dx;
                const double y = static_cast<double>(j) * grid.dy;
                host_temperature[static_cast<std::size_t>(j) * grid.nx + i] =
                    1.0 - y + 0.01 * std::sin(M_PI * x / 2.0) * std::sin(M_PI * y);
            }
        }

        // 温度推进采用双缓冲：读取旧温度，写入新温度，避免原地更新引入方向偏差。
        geodynamics::DeviceBuffer<double> temperature(grid.cell_count());
        geodynamics::DeviceBuffer<double> next_temperature(grid.cell_count());
        temperature.copy_from_host(host_temperature.data(), grid.cell_count());
        geodynamics::apply_temperature_boundaries(temperature.data(), grid, thermal);

        // 当前仅演示纯扩散的固定步数；后续这里会加入 Stokes 求解和 CFL 控制。
        constexpr int steps = 20;
        for (int step = 0; step < steps; ++step) {
            geodynamics::advance_temperature_diffusion(
                temperature.data(), next_temperature.data(), grid, thermal);
            geodynamics::apply_temperature_boundaries(next_temperature.data(), grid, thermal);
            std::swap(temperature, next_temperature);
        }

        // 该归约结果可作为最基本的数值状态检查。
        const double maximum_temperature = geodynamics::max_abs_device(temperature.data(), grid.cell_count());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::cout << "CUDA device: " << properties.name << '\n';
        std::cout << "Grid: " << grid.nx << " x " << grid.ny << '\n';
        std::cout << "Diffusion steps: " << steps << '\n';
        std::cout << "Maximum absolute temperature: " << maximum_temperature << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << '\n';
        return 1;
    }
}
