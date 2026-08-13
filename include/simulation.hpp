#pragma once

#include "grid.hpp"

#include <cstddef>

namespace geodynamics {

// 当前温度方程所需的物理参数和上下边界温度。
struct ThermalParameters {
    double diffusivity;        // 热扩散系数 kappa。
    double dt;                 // 显式时间推进的物理时间步长。
    double bottom_temperature; // j = 0 的固定温度。
    double top_temperature;    // j = ny - 1 的固定温度。
};

// 指定解析旋涡速度场的参数。该场由流函数导出，连续意义下满足 div(u) = 0。
// strength 的量纲为长度平方/时间；速度由它对空间的导数得到。
struct VortexParameters {
    double strength;
};

// 每个输出步记录的温度方程稳定性和有效性诊断。
struct TemperatureDiagnostics {
    double minimum_temperature;
    double maximum_temperature;
    double mean_temperature;
    double mean_squared_temperature;
    double advection_cfl;
    double diffusion_cfl;
    bool is_finite;
};

// 在设备端施加温度边界：上下固定温度，左右使用零法向梯度（绝热）条件。
void apply_temperature_boundaries(double* temperature, const Grid2D& grid,
                                  const ThermalParameters& parameters);

// 在设备端完成一步显式热扩散。输入和输出必须是不同的设备数组。
void advance_temperature_diffusion(const double* temperature, double* next_temperature,
                                   const Grid2D& grid, const ThermalParameters& parameters);

// 完成一步一阶迎风平流-扩散。旋涡速度在单元中心解析计算，不需要存储 u/v 数组。
void advance_temperature_advection_diffusion(const double* temperature, double* next_temperature,
                                             const Grid2D& grid, const ThermalParameters& thermal,
                                             const VortexParameters& vortex);

// 对设备数组进行归约，返回 max(abs(values[i]))，用于后续残差诊断。
double max_abs_device(const double* values, std::size_t count);

// 在 GPU 上计算温度范围、CFL 数和 NaN/Inf 标志，仅将少量标量返回 CPU。
TemperatureDiagnostics compute_temperature_diagnostics(const double* temperature, const Grid2D& grid,
                                                       const ThermalParameters& thermal,
                                                       const VortexParameters& vortex);

}  // namespace geodynamics
