#pragma once

#include "grid.hpp"

namespace geodynamics {

// 在每个单元中心计算 div(u) = (u_right-u_left)/dx + (v_top-v_bottom)/dy。
// u 的设备数组大小为 grid.u_count()，v 为 grid.v_count()，输出为 grid.cell_count()。
void compute_mac_divergence(const double* u, const double* v, double* divergence,
                            const Grid2D& grid);

// 在交错速度面计算压力梯度。外边界面暂设为零，具体压力边界条件将在 Stokes 阶段确定。
// gradient_u 大小为 grid.u_count()，gradient_v 大小为 grid.v_count()。
void compute_pressure_gradient(const double* pressure, double* gradient_u, double* gradient_v,
                               const Grid2D& grid);

// 常黏度伪瞬态 Stokes 参数。两个松弛参数已吸收数值密度和体积模量，
// 由显式稳定性约束决定：lambda_u <= h^2/(2 d eta)、lambda_p * lambda_u <= h^2/d。
struct StokesParameters {
    double viscosity;            // 恒定黏度 eta。
    double momentum_relaxation;  // 动量更新的伪时间步 lambda_u。
    double pressure_relaxation;  // 连续性/压力更新的伪时间步 lambda_p。
};

// 执行一步伪瞬态 Stokes 更新（动量 + 连续性），速度边界为全自由滑移：
// 法向速度在边界置零，切向速度在边界用镜像满足零切应力（法向导数为零）。
// fx 尺寸为 grid.u_count()，fy 尺寸为 grid.v_count()；输入与输出数组必须互不相同。
void advance_stokes_pseudotransient_step(const double* u, const double* v, const double* pressure,
                                        const double* fx, const double* fy,
                                        double* u_next, double* v_next, double* pressure_next,
                                        const Grid2D& grid, const StokesParameters& parameters);

// 去除压力场均值，固定压力规范（全自由滑移下压力仅确定到常数）。
void subtract_mean_pressure(double* pressure, const Grid2D& grid);

}  // namespace geodynamics
