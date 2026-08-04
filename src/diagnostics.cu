#include "cuda_utils.hpp"
#include "simulation.hpp"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <cmath>

namespace geodynamics {
namespace {

// Thrust 归约使用的逐元素变换：先取绝对值，再由 maximum 求最大值。
struct AbsoluteValue {
    __host__ __device__ double operator()(double value) const {
        return value < 0.0 ? -value : value;
    }
};

}  // namespace

double max_abs_device(const double* values, std::size_t count) {
    // 归约全过程留在 GPU，避免为一个诊断量复制整块数组回 CPU。
    thrust::device_ptr<const double> begin(values);
    return thrust::transform_reduce(thrust::device, begin, begin + count, AbsoluteValue{}, 0.0,
                                    thrust::maximum<double>());
}

}  // namespace geodynamics
