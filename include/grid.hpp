#pragma once

#include <cstddef>
#include <stdexcept>

namespace geodynamics {

// 二维规则网格。标量场按 [ny][nx] 行优先方式压平为一维数组。
struct Grid2D {
    int nx;      // x 方向单元数，对应一维索引中的列数。
    int ny;      // y 方向单元数，对应一维索引中的行数。
    double dx;   // x 方向单元间距。
    double dy;   // y 方向单元间距。

    // 至少保留一层内部单元，才能计算五点差分拉普拉斯算子。
    Grid2D(int nx_in, int ny_in, double dx_in, double dy_in)
        : nx(nx_in), ny(ny_in), dx(dx_in), dy(dy_in) {
        if (nx < 3 || ny < 3) {
            throw std::invalid_argument("Grid2D needs at least three cells in each direction");
        }
        if (dx <= 0.0 || dy <= 0.0) {
            throw std::invalid_argument("Grid2D spacing must be positive");
        }
    }

    // 单元中心场 T、p、eta 的总元素数。
    [[nodiscard]] std::size_t cell_count() const {
        return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    }

    // MAC 网格中 x 方向速度 u 位于竖直面，因此 x 方向比单元数多一列。
    [[nodiscard]] std::size_t u_count() const {
        return static_cast<std::size_t>(nx + 1) * static_cast<std::size_t>(ny);
    }

    // MAC 网格中 y 方向速度 v 位于水平面，因此 y 方向比单元数多一行。
    [[nodiscard]] std::size_t v_count() const {
        return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny + 1);
    }
};

}  // namespace geodynamics
