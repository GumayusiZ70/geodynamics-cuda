#include "output.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace geodynamics {
namespace {

// 将归一化温度映射为蓝-青-黄-红渐变，低温为蓝，高温为红。
std::array<unsigned char, 3> temperature_color(double normalized_temperature) {
    const double value = std::clamp(normalized_temperature, 0.0, 1.0);
    if (value < 0.25) {
        const double local = value / 0.25;
        return {0, static_cast<unsigned char>(255.0 * local), 255};
    }
    if (value < 0.5) {
        const double local = (value - 0.25) / 0.25;
        return {0, 255, static_cast<unsigned char>(255.0 * (1.0 - local))};
    }
    if (value < 0.75) {
        const double local = (value - 0.5) / 0.25;
        return {static_cast<unsigned char>(255.0 * local), 255, 0};
    }
    const double local = (value - 0.75) / 0.25;
    return {255, static_cast<unsigned char>(255.0 * (1.0 - local)), 0};
}

void ensure_parent_directory(const std::string& path) {
    const std::filesystem::path file_path(path);
    if (!file_path.parent_path().empty()) {
        std::filesystem::create_directories(file_path.parent_path());
    }
}

// 生成 PNG/PPM 共用的 RGB 像素。j 从 ny-1 向下写，使图片顶部对应物理域顶部。
std::vector<unsigned char> make_rgb_pixels(const std::vector<double>& temperature, const Grid2D& grid,
                                           double top_temperature, double bottom_temperature) {
    std::vector<unsigned char> pixels;
    pixels.reserve(grid.cell_count() * 3);
    for (int j = grid.ny - 1; j >= 0; --j) {
        for (int i = 0; i < grid.nx; ++i) {
            const std::size_t index = static_cast<std::size_t>(j) * grid.nx + i;
            const double normalized = (temperature[index] - top_temperature) /
                                      (bottom_temperature - top_temperature);
            const auto color = temperature_color(normalized);
            pixels.insert(pixels.end(), color.begin(), color.end());
        }
    }
    return pixels;
}

// PNG 使用大端序存储 32 位长度和 CRC。
void write_uint32_be(std::ofstream& output, std::uint32_t value) {
    const unsigned char bytes[4] = {
        static_cast<unsigned char>((value >> 24) & 0xffU),
        static_cast<unsigned char>((value >> 16) & 0xffU),
        static_cast<unsigned char>((value >> 8) & 0xffU),
        static_cast<unsigned char>(value & 0xffU),
    };
    output.write(reinterpret_cast<const char*>(bytes), 4);
}

// PNG chunk 的标准 CRC-32 实现；小型输出文件无需引入额外压缩库。
std::uint32_t crc32(const unsigned char* data, std::size_t size) {
    std::uint32_t crc = 0xffffffffU;
    for (std::size_t index = 0; index < size; ++index) {
        crc ^= data[index];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc & 1U) != 0U ? (crc >> 1U) ^ 0xedb88320U : crc >> 1U;
        }
    }
    return crc ^ 0xffffffffU;
}

std::uint32_t adler32(const std::vector<unsigned char>& data) {
    constexpr std::uint32_t modulus = 65521U;
    std::uint32_t a = 1U;
    std::uint32_t b = 0U;
    for (const unsigned char value : data) {
        a = (a + value) % modulus;
        b = (b + a) % modulus;
    }
    return (b << 16U) | a;
}

void write_png_chunk(std::ofstream& output, const char type[4],
                     const std::vector<unsigned char>& data) {
    write_uint32_be(output, static_cast<std::uint32_t>(data.size()));
    output.write(type, 4);
    if (!data.empty()) {
        output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }

    std::vector<unsigned char> crc_input(type, type + 4);
    crc_input.insert(crc_input.end(), data.begin(), data.end());
    write_uint32_be(output, crc32(crc_input.data(), crc_input.size()));
}

// 将 PNG 的原始扫描线封装进 zlib。使用 DEFLATE 的 stored block，避免额外依赖。
std::vector<unsigned char> zlib_store(const std::vector<unsigned char>& raw_scanlines) {
    std::vector<unsigned char> compressed;
    compressed.reserve(raw_scanlines.size() + 32);
    compressed.push_back(0x78U);
    compressed.push_back(0x01U);

    std::size_t offset = 0;
    while (offset < raw_scanlines.size()) {
        const std::size_t block_size = std::min<std::size_t>(65535, raw_scanlines.size() - offset);
        const bool is_final = offset + block_size == raw_scanlines.size();
        compressed.push_back(is_final ? 0x01U : 0x00U);
        compressed.push_back(static_cast<unsigned char>(block_size & 0xffU));
        compressed.push_back(static_cast<unsigned char>((block_size >> 8U) & 0xffU));
        const std::uint16_t inverted_size = static_cast<std::uint16_t>(~block_size);
        compressed.push_back(static_cast<unsigned char>(inverted_size & 0xffU));
        compressed.push_back(static_cast<unsigned char>((inverted_size >> 8U) & 0xffU));
        compressed.insert(compressed.end(), raw_scanlines.begin() + static_cast<std::ptrdiff_t>(offset),
                          raw_scanlines.begin() + static_cast<std::ptrdiff_t>(offset + block_size));
        offset += block_size;
    }

    const std::uint32_t checksum = adler32(raw_scanlines);
    compressed.push_back(static_cast<unsigned char>((checksum >> 24U) & 0xffU));
    compressed.push_back(static_cast<unsigned char>((checksum >> 16U) & 0xffU));
    compressed.push_back(static_cast<unsigned char>((checksum >> 8U) & 0xffU));
    compressed.push_back(static_cast<unsigned char>(checksum & 0xffU));
    return compressed;
}

}  // namespace

void write_temperature_csv(const std::string& path, const std::vector<double>& temperature,
                           const Grid2D& grid) {
    if (temperature.size() != grid.cell_count()) {
        throw std::invalid_argument("温度数组大小与网格不一致，无法写入 CSV");
    }
    ensure_parent_directory(path);
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("无法打开温度 CSV 输出文件: " + path);
    }

    output << "x,y,temperature\n";
    output << std::setprecision(17);
    for (int j = 0; j < grid.ny; ++j) {
        for (int i = 0; i < grid.nx; ++i) {
            const std::size_t index = static_cast<std::size_t>(j) * grid.nx + i;
            output << i * grid.dx << ',' << j * grid.dy << ',' << temperature[index] << '\n';
        }
    }
}

void write_temperature_ppm(const std::string& path, const std::vector<double>& temperature,
                           const Grid2D& grid, double top_temperature, double bottom_temperature) {
    if (temperature.size() != grid.cell_count()) {
        throw std::invalid_argument("温度数组大小与网格不一致，无法写入 PPM");
    }
    if (bottom_temperature <= top_temperature) {
        throw std::invalid_argument("PPM 温度颜色范围必须满足 bottom_temperature > top_temperature");
    }
    ensure_parent_directory(path);
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("无法打开温度 PPM 输出文件: " + path);
    }

    // PPM 图像的首行是顶部，因此 RGB 像素按物理坐标从上到下排列。
    output << "P6\n" << grid.nx << ' ' << grid.ny << "\n255\n";
    const auto pixels = make_rgb_pixels(temperature, grid, top_temperature, bottom_temperature);
    output.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
}

void write_temperature_png(const std::string& path, const std::vector<double>& temperature,
                           const Grid2D& grid, double top_temperature, double bottom_temperature) {
    if (temperature.size() != grid.cell_count()) {
        throw std::invalid_argument("温度数组大小与网格不一致，无法写入 PNG");
    }
    if (bottom_temperature <= top_temperature) {
        throw std::invalid_argument("PNG 温度颜色范围必须满足 bottom_temperature > top_temperature");
    }
    ensure_parent_directory(path);
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("无法打开温度 PNG 输出文件: " + path);
    }

    // PNG 签名、IHDR、IDAT、IEND 四部分构成最小有效真彩色 PNG。
    constexpr unsigned char signature[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    output.write(reinterpret_cast<const char*>(signature), 8);
    std::vector<unsigned char> ihdr(13, 0);
    ihdr[0] = static_cast<unsigned char>((grid.nx >> 24) & 0xff);
    ihdr[1] = static_cast<unsigned char>((grid.nx >> 16) & 0xff);
    ihdr[2] = static_cast<unsigned char>((grid.nx >> 8) & 0xff);
    ihdr[3] = static_cast<unsigned char>(grid.nx & 0xff);
    ihdr[4] = static_cast<unsigned char>((grid.ny >> 24) & 0xff);
    ihdr[5] = static_cast<unsigned char>((grid.ny >> 16) & 0xff);
    ihdr[6] = static_cast<unsigned char>((grid.ny >> 8) & 0xff);
    ihdr[7] = static_cast<unsigned char>(grid.ny & 0xff);
    ihdr[8] = 8;  // 每个通道 8 bit。
    ihdr[9] = 2;  // 真彩色 RGB。
    write_png_chunk(output, "IHDR", ihdr);

    const auto pixels = make_rgb_pixels(temperature, grid, top_temperature, bottom_temperature);
    std::vector<unsigned char> scanlines;
    const std::size_t row_bytes = static_cast<std::size_t>(grid.nx) * 3;
    scanlines.reserve(static_cast<std::size_t>(grid.ny) * (row_bytes + 1));
    for (int j = 0; j < grid.ny; ++j) {
        scanlines.push_back(0);  // PNG filter type 0：不做行内预测，便于验证。
        const auto begin = pixels.begin() + static_cast<std::ptrdiff_t>(j) * row_bytes;
        scanlines.insert(scanlines.end(), begin, begin + static_cast<std::ptrdiff_t>(row_bytes));
    }
    write_png_chunk(output, "IDAT", zlib_store(scanlines));
    write_png_chunk(output, "IEND", {});
}

void write_diagnostics_csv_header(const std::string& path) {
    ensure_parent_directory(path);
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("无法打开诊断 CSV 输出文件: " + path);
    }
    output << "step,time,min_temperature,max_temperature,mean_temperature,mean_squared_temperature,"
              "advection_cfl,diffusion_cfl,is_finite\n";
}

void append_temperature_diagnostics_csv(const std::string& path, int step, double time,
                                        const TemperatureDiagnostics& diagnostics) {
    std::ofstream output(path, std::ios::app);
    if (!output) {
        throw std::runtime_error("无法追加诊断 CSV 输出文件: " + path);
    }
    output << std::setprecision(17) << step << ',' << time << ',' << diagnostics.minimum_temperature << ','
           << diagnostics.maximum_temperature << ',' << diagnostics.mean_temperature << ','
           << diagnostics.mean_squared_temperature << ',' << diagnostics.advection_cfl << ','
           << diagnostics.diffusion_cfl << ',' << (diagnostics.is_finite ? 1 : 0) << '\n';
}

}  // namespace geodynamics
