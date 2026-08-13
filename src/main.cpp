#include "cuda_utils.hpp"
#include "output.hpp"
#include "simulation.hpp"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
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
        // 参数特意选在显式扩散和平流稳定范围内；诊断会在每个输出步报告 CFL。
        const geodynamics::ThermalParameters thermal{1.0e-3, 1.0e-3, 1.0, 0.0};
        const geodynamics::VortexParameters vortex{0.1};

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

        constexpr int steps = 1000;
        constexpr int output_interval = 200;
        const std::filesystem::path output_directory("output");
        const std::filesystem::path diagnostics_path = output_directory / "diagnostics.csv";
        std::filesystem::create_directories(output_directory);
        geodynamics::write_diagnostics_csv_header(diagnostics_path.string());

        // 每个快照同时提供通用 CSV 和可直接打开的彩色 PPM 图像。
        auto write_snapshot = [&](int step) {
            std::vector<double> host_snapshot(grid.cell_count());
            temperature.copy_to_host(host_snapshot.data(), grid.cell_count());
            std::ostringstream stem_builder;
            stem_builder << "temperature_" << std::setfill('0') << std::setw(6) << step;
            const std::string stem = stem_builder.str();
            geodynamics::write_temperature_csv((output_directory / (stem + ".csv")).string(),
                                               host_snapshot, grid);
            geodynamics::write_temperature_ppm((output_directory / (stem + ".ppm")).string(),
                                               host_snapshot, grid, thermal.top_temperature,
                                               thermal.bottom_temperature);
            geodynamics::write_temperature_png((output_directory / (stem + ".png")).string(),
                                               host_snapshot, grid, thermal.top_temperature,
                                               thermal.bottom_temperature);

            const auto diagnostics = geodynamics::compute_temperature_diagnostics(
                temperature.data(), grid, thermal, vortex);
            geodynamics::append_temperature_diagnostics_csv(
                diagnostics_path.string(), step, step * thermal.dt, diagnostics);
            std::cout << "步数=" << std::setw(4) << step
                      << " 时间=" << std::fixed << std::setprecision(3) << step * thermal.dt
                      << " Tmin=" << std::setprecision(6) << diagnostics.minimum_temperature
                      << " Tmax=" << diagnostics.maximum_temperature
                      << " Tmean=" << diagnostics.mean_temperature
                      << " T2mean=" << diagnostics.mean_squared_temperature
                      << " CFL_adv=" << diagnostics.advection_cfl
                      << " CFL_diff=" << diagnostics.diffusion_cfl
                      << " 有限=" << (diagnostics.is_finite ? "是" : "否") << '\n';
        };

        write_snapshot(0);
        for (int step = 0; step < steps; ++step) {
            geodynamics::advance_temperature_advection_diffusion(
                temperature.data(), next_temperature.data(), grid, thermal, vortex);
            geodynamics::apply_temperature_boundaries(next_temperature.data(), grid, thermal);
            std::swap(temperature, next_temperature);
            if ((step + 1) % output_interval == 0) {
                write_snapshot(step + 1);
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        std::cout << "CUDA 设备: " << properties.name << '\n';
        std::cout << "网格: " << grid.nx << " x " << grid.ny << '\n';
        std::cout << "已完成平流-扩散步数: " << steps << '\n';
        std::cout << "输出目录: " << output_directory << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << '\n';
        return 1;
    }
}
