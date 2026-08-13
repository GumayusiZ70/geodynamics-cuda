#pragma once

#include "grid.hpp"

#include <cstddef>

namespace geodynamics {

// MAC 交错网格的一维数组索引约定。
// 标量场 p/T/eta 为 [ny][nx]；u 为 [ny][nx+1]；v 为 [ny+1][nx]。
inline std::size_t scalar_index(const Grid2D& grid, int i, int j) {
    return static_cast<std::size_t>(j) * grid.nx + i;
}

inline std::size_t u_index(const Grid2D& grid, int i, int j) {
    return static_cast<std::size_t>(j) * (grid.nx + 1) + i;
}

inline std::size_t v_index(const Grid2D& grid, int i, int j) {
    return static_cast<std::size_t>(j) * grid.nx + i;
}

}  // namespace geodynamics
