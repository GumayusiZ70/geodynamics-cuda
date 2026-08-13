#include "cuda_utils.hpp"
#include "mac_grid.hpp"
#include "stokes.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_constant(const std::vector<double>& values, double expected, const char* label) {
    double maximum_error = 0.0;
    for (const double value : values) {
        maximum_error = std::max(maximum_error, std::abs(value - expected));
    }
    if (maximum_error > 1.0e-12) {
        throw std::runtime_error(std::string(label) + " 最大误差为 " + std::to_string(maximum_error));
    }
}

void require_near(double value, double expected, const char* label) {
    const double error = std::abs(value - expected);
    if (error > 1.0e-12) {
        throw std::runtime_error(std::string(label) + " 误差为 " + std::to_string(error));
    }
}

}  // namespace

int main() {
    try {
        const geodynamics::Grid2D grid(7, 5, 0.25, 0.5);

        // 构造 u=x、v=2y，离散散度应精确等于 1+2=3。
        std::vector<double> host_u(grid.u_count());
        std::vector<double> host_v(grid.v_count());
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i <= grid.nx; ++i) {
                host_u[geodynamics::u_index(grid, i, j)] = i * grid.dx;
            }
        }
        for (int j = 0; j <= grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                host_v[geodynamics::v_index(grid, i, j)] = 2.0 * j * grid.dy;
            }
        }

        geodynamics::DeviceBuffer<double> u(grid.u_count());
        geodynamics::DeviceBuffer<double> v(grid.v_count());
        geodynamics::DeviceBuffer<double> divergence(grid.cell_count());
        u.copy_from_host(host_u.data(), grid.u_count());
        v.copy_from_host(host_v.data(), grid.v_count());
        geodynamics::compute_mac_divergence(u.data(), v.data(), divergence.data(), grid);
        std::vector<double> host_divergence(grid.cell_count());
        divergence.copy_to_host(host_divergence.data(), grid.cell_count());
        require_constant(host_divergence, 3.0, "MAC 散度");

        // 线性压力 p=3x-2y+5 的内部面梯度应精确为 dp/dx=3、dp/dy=-2。
        std::vector<double> host_pressure(grid.cell_count());
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                host_pressure[geodynamics::scalar_index(grid, i, j)] =
                    3.0 * i * grid.dx - 2.0 * j * grid.dy + 5.0;
            }
        }

        geodynamics::DeviceBuffer<double> pressure(grid.cell_count());
        geodynamics::DeviceBuffer<double> gradient_u(grid.u_count());
        geodynamics::DeviceBuffer<double> gradient_v(grid.v_count());
        pressure.copy_from_host(host_pressure.data(), grid.cell_count());
        geodynamics::compute_pressure_gradient(pressure.data(), gradient_u.data(), gradient_v.data(), grid);
        std::vector<double> host_gradient_u(grid.u_count());
        std::vector<double> host_gradient_v(grid.v_count());
        gradient_u.copy_to_host(host_gradient_u.data(), grid.u_count());
        gradient_v.copy_to_host(host_gradient_v.data(), grid.v_count());

        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i <= grid.nx; ++i) {
                const double expected = (i == 0 || i == grid.nx) ? 0.0 : 3.0;
                require_near(host_gradient_u[geodynamics::u_index(grid, i, j)], expected,
                             "u 面压力梯度");
            }
        }
        for (int j = 0; j <= grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                const double expected = (j == 0 || j == grid.ny) ? 0.0 : -2.0;
                require_near(host_gradient_v[geodynamics::v_index(grid, i, j)], expected,
                             "v 面压力梯度");
            }
        }

        std::cout << "MAC 网格散度和压力梯度测试通过\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MAC 网格测试失败: " << error.what() << '\n';
        return 1;
    }
}
