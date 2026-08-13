#include "cuda_utils.hpp"
#include "mac_grid.hpp"
#include "stokes.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

// 离散流函数在角点 (i,j) 的值。用它对交错速度做差，可保证离散散度恒为零（望远镜求和消去）。
double stream_function(int i, int j, int nx, int ny) {
    return std::sin(pi * i / nx) * std::sin(pi * j / ny);
}

// 与 GPU momentum_u_kernel 完全一致的离散拉普拉斯（含自由滑移镜像）。
double lap_u_cpu(const std::vector<double>& u, int i, int j, int nx, int ny, double dx, double dy) {
    const int face = j * (nx + 1) + i;
    const double center = u[face];
    const double lap_x = (u[face - 1] - 2.0 * center + u[face + 1]) / (dx * dx);
    const double below = (j > 0) ? u[face - (nx + 1)] : center;
    const double above = (j < ny - 1) ? u[face + (nx + 1)] : center;
    const double lap_y = (below - 2.0 * center + above) / (dy * dy);
    return lap_x + lap_y;
}

// 与 GPU momentum_v_kernel 完全一致的离散拉普拉斯。
double lap_v_cpu(const std::vector<double>& v, int i, int j, int nx, int ny, double dx, double dy) {
    const int face = j * nx + i;
    const double center = v[face];
    const double lap_y = (v[face - nx] - 2.0 * center + v[face + nx]) / (dy * dy);
    const double left = (i > 0) ? v[face - 1] : center;
    const double right = (i < nx - 1) ? v[face + 1] : center;
    const double lap_x = (left - 2.0 * center + right) / (dx * dx);
    return lap_x + lap_y;
}

double max_error(const std::vector<double>& a, const std::vector<double>& b) {
    double error = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) {
        error = std::max(error, std::abs(a[k] - b[k]));
    }
    return error;
}

}  // namespace

int main() {
    try {
        const int nx = 20;
        const int ny = 10;
        const double dx = 1.0;
        const double dy = 1.0;
        const geodynamics::Grid2D grid(nx, ny, dx, dy);
        const double viscosity = 1.0;

        // 参考速度场由离散流函数导出，天然满足全自由滑移（法向零、切向零法向导数）与离散无散。
        std::vector<double> u_ref(grid.u_count());
        std::vector<double> v_ref(grid.v_count());
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                u_ref[geodynamics::u_index(grid, i, j)] =
                    (stream_function(i, j + 1, nx, ny) - stream_function(i, j, nx, ny)) / dy;
            }
        }
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                v_ref[geodynamics::v_index(grid, i, j)] =
                    -(stream_function(i + 1, j, nx, ny) - stream_function(i, j, nx, ny)) / dx;
            }
        }

        // 参考压力取任意光滑场，再去均值以匹配求解器的零均值规范。
        std::vector<double> p_ref(grid.cell_count());
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                p_ref[geodynamics::scalar_index(grid, i, j)] =
                    std::cos(pi * (i + 0.5) / nx) * std::cos(pi * (j + 0.5) / ny);
            }
        }
        double pressure_mean = 0.0;
        for (double value : p_ref) {
            pressure_mean += value;
        }
        pressure_mean /= static_cast<double>(p_ref.size());
        for (double& value : p_ref) {
            value -= pressure_mean;
        }

        // 离散 MMS：用与 GPU 相同的离散算子从参考场构造体积力 f = grad(p) - eta*lap(u)，
        // 使参考场成为离散系统 eta*lap(u) - grad(p) + f = 0 的精确解。
        std::vector<double> fx(grid.u_count(), 0.0);
        std::vector<double> fy(grid.v_count(), 0.0);
        for (int j = 0; j < ny; ++j) {
            for (int i = 1; i < nx; ++i) {  // 内部 u 面。
                const double lap = lap_u_cpu(u_ref, i, j, nx, ny, dx, dy);
                const double grad_p = (p_ref[geodynamics::scalar_index(grid, i, j)] -
                                       p_ref[geodynamics::scalar_index(grid, i - 1, j)]) / dx;
                fx[geodynamics::u_index(grid, i, j)] = grad_p - viscosity * lap;
            }
        }
        for (int j = 1; j < ny; ++j) {  // 内部 v 面。
            for (int i = 0; i < nx; ++i) {
                const double lap = lap_v_cpu(v_ref, i, j, nx, ny, dx, dy);
                const double grad_p = (p_ref[geodynamics::scalar_index(grid, i, j)] -
                                       p_ref[geodynamics::scalar_index(grid, i, j - 1)]) / dy;
                fy[geodynamics::v_index(grid, i, j)] = grad_p - viscosity * lap;
            }
        }

        // 初始速度、压力置零，仅保留体积力。
        geodynamics::DeviceBuffer<double> u(grid.u_count());
        geodynamics::DeviceBuffer<double> v(grid.v_count());
        geodynamics::DeviceBuffer<double> pressure(grid.cell_count());
        geodynamics::DeviceBuffer<double> u_next(grid.u_count());
        geodynamics::DeviceBuffer<double> v_next(grid.v_count());
        geodynamics::DeviceBuffer<double> pressure_next(grid.cell_count());
        geodynamics::DeviceBuffer<double> fx_device(grid.u_count());
        geodynamics::DeviceBuffer<double> fy_device(grid.v_count());
        CUDA_CHECK(cudaMemset(u.data(), 0, u.size() * sizeof(double)));
        CUDA_CHECK(cudaMemset(v.data(), 0, v.size() * sizeof(double)));
        CUDA_CHECK(cudaMemset(pressure.data(), 0, pressure.size() * sizeof(double)));
        fx_device.copy_from_host(fx.data(), grid.u_count());
        fy_device.copy_from_host(fy.data(), grid.v_count());

        // 松弛参数满足显式稳定性：lambda_u <= h^2/(2 d eta)、lambda_p * lambda_u <= h^2/d。
        const geodynamics::StokesParameters params{viscosity, 0.2, 0.5};

        constexpr int max_iterations = 200000;
        constexpr int check_interval = 500;
        const double tolerance = 1.0e-8;

        std::vector<double> host_u(grid.u_count());
        std::vector<double> host_v(grid.v_count());
        std::vector<double> host_p(grid.cell_count());

        bool converged = false;
        int iteration = 0;
        for (; iteration < max_iterations; ++iteration) {
            geodynamics::advance_stokes_pseudotransient_step(
                u.data(), v.data(), pressure.data(), fx_device.data(), fy_device.data(),
                u_next.data(), v_next.data(), pressure_next.data(), grid, params);
            std::swap(u, u_next);
            std::swap(v, v_next);
            std::swap(pressure, pressure_next);

            if ((iteration + 1) % check_interval == 0) {
                u.copy_to_host(host_u.data(), grid.u_count());
                v.copy_to_host(host_v.data(), grid.v_count());
                pressure.copy_to_host(host_p.data(), grid.cell_count());
                const double error = std::max({max_error(host_u, u_ref),
                                               max_error(host_v, v_ref),
                                               max_error(host_p, p_ref)});
                std::cout << "迭代 " << (iteration + 1) << " 最大误差 " << error << '\n';
                if (error < tolerance) {
                    converged = true;
                    break;
                }
            }
        }

        if (!converged) {
            u.copy_to_host(host_u.data(), grid.u_count());
            v.copy_to_host(host_v.data(), grid.v_count());
            pressure.copy_to_host(host_p.data(), grid.cell_count());
            const double error = std::max({max_error(host_u, u_ref),
                                           max_error(host_v, v_ref),
                                           max_error(host_p, p_ref)});
            throw std::runtime_error("Stokes 伪瞬态在 " + std::to_string(max_iterations) +
                                     " 步内未收敛，最终最大误差 " + std::to_string(error));
        }

        std::cout << "Stokes 伪瞬态制造解测试通过\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Stokes MMS test failed: " << error.what() << '\n';
        return 1;
    }
}
