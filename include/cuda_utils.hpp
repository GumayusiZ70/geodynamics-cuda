#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace geodynamics {

// 将 CUDA API 的错误转换为带有源文件、行号和原始表达式的 C++ 异常。
// 这样 cudaMalloc、cudaMemcpy 或 kernel 启动失败时，调用者能定位到具体代码行。
inline void cuda_check(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA failure at ") + file + ":" + std::to_string(line) +
            " while evaluating " + expression + ": " + cudaGetErrorString(status));
    }
}

// 统一包装 CUDA 调用，自动记录调用表达式及其所在位置。
#define CUDA_CHECK(expression) ::geodynamics::cuda_check((expression), #expression, __FILE__, __LINE__)

// 简单的设备端一维数组所有者。它只管理内存，不管理数值含义。
// 标量场和 MAC 网格数组均可用该类存储，索引规则由调用方决定。
template <typename T>
class DeviceBuffer {
public:
    // 申请 count 个元素的 GPU 全局内存；零长度缓冲区保持空指针。
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ > 0) {
            CUDA_CHECK(cudaMalloc(&data_, count_ * sizeof(T)));
        }
    }

    // 析构时释放设备内存。析构函数不能抛异常，因此此处不再调用 CUDA_CHECK。
    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    // 设备指针只能有一个所有者，禁止复制，允许转移所有权。
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                cudaFree(data_);
            }
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    // 仅返回裸设备指针，kernel 启动和 cudaMemcpy 使用该指针。
    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }

    // 从 CPU 内存复制到 GPU；复制前检查元素数量，避免越过分配边界。
    void copy_from_host(const T* source, std::size_t count) {
        ensure_count(count);
        CUDA_CHECK(cudaMemcpy(data_, source, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    // 从 GPU 内存复制到 CPU；cudaMemcpy 会等待此前对该默认流的操作完成。
    void copy_to_host(T* destination, std::size_t count) const {
        ensure_count(count);
        CUDA_CHECK(cudaMemcpy(destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

private:
    // 所有主机/设备复制都复用此检查，防止静默的内存越界。
    void ensure_count(std::size_t count) const {
        if (count > count_) {
            throw std::out_of_range("DeviceBuffer copy exceeds allocated size");
        }
    }

    T* data_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace geodynamics
