cmake_minimum_required(VERSION 3.22)

# 本项目同时编译普通 C++ 主程序和 CUDA kernel。
project(geodynamics_cuda LANGUAGES CXX CUDA)

# 统一两端语言标准，避免不同开发者的默认编译器标准不一致。
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# 核心库集中保存数值模块，主程序和测试程序均链接此库。
add_library(geodynamics_core STATIC
    src/boundary.cu
    src/diagnostics.cu
    src/multigrid.cu
    src/stokes_pt.cu
    src/temperature.cu
)

target_include_directories(geodynamics_core PUBLIC include)
target_link_libraries(geodynamics_core PUBLIC CUDA::cudart)
# 未来一个数值模块可能含多个 CUDA 编译单元，因此启用设备端链接。
set_target_properties(geodynamics_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 演示程序：当前执行二维温度扩散，后续扩展为完整时间步循环。
add_executable(geodynamics src/main.cpp)
target_link_libraries(geodynamics PRIVATE geodynamics_core)

# CTest 用于把 CPU/GPU 数值对照纳入统一测试入口。
include(CTest)
if(BUILD_TESTING)
    add_executable(test_temperature tests/test_temperature.cpp)
    target_link_libraries(test_temperature PRIVATE geodynamics_core)
    add_test(NAME temperature_diffusion_step COMMAND test_temperature)
endif()
