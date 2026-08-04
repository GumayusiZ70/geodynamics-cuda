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

// 在设备端施加温度边界：上下固定温度，左右使用零法向梯度（绝热）条件。
void apply_temperature_boundaries(double* temperature, const Grid2D& grid,
                                  const ThermalParameters& parameters);

// 在设备端完成一步显式热扩散。输入和输出必须是不同的设备数组。
void advance_temperature_diffusion(const double* temperature, double* next_temperature,
                                   const Grid2D& grid, const ThermalParameters& parameters);

// 对设备数组进行归约，返回 max(abs(values[i]))，用于后续残差诊断。
double max_abs_device(const double* values, std::size_t count);

}  // namespace geodynamics
