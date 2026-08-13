#pragma once

#include "simulation.hpp"

#include <string>
#include <vector>

namespace geodynamics {

// 将单元中心温度以 x,y,temperature 三列写入 CSV，便于 Python、ParaView 等工具读取。
void write_temperature_csv(const std::string& path, const std::vector<double>& temperature,
                           const Grid2D& grid);

// 将温度场写为标准二进制 PPM 彩色图。颜色范围固定为 [top_temperature, bottom_temperature]。
void write_temperature_ppm(const std::string& path, const std::vector<double>& temperature,
                           const Grid2D& grid, double top_temperature, double bottom_temperature);

// 无外部图像库地写出标准 PNG，便于在 Windows、浏览器和 VS Code 中直接查看。
void write_temperature_png(const std::string& path, const std::vector<double>& temperature,
                           const Grid2D& grid, double top_temperature, double bottom_temperature);

// 初始化诊断 CSV。每一次运行都覆盖旧日志，确保该日志只描述当前一次模拟。
void write_diagnostics_csv_header(const std::string& path);

// 向诊断 CSV 追加一个输出步的稳定性、温度范围和能量代理量。
void append_temperature_diagnostics_csv(const std::string& path, int step, double time,
                                        const TemperatureDiagnostics& diagnostics);

}  // namespace geodynamics
