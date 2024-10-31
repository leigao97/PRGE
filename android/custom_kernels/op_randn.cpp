/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

#include <cstdint>
#include <random>

namespace torch {
namespace executor {
namespace native {

using exec_aten::Tensor;

namespace {

bool check_sizes(
    exec_aten::ArrayRef<int64_t> size_int64_t,
    exec_aten::ArrayRef<int32_t> size_int32_t) {
  ET_LOG_AND_RETURN_IF_FALSE(size_int64_t.size() == size_int32_t.size());
  for (int i = 0; i < size_int64_t.size(); i++) {
    ET_LOG_AND_RETURN_IF_FALSE(((int64_t)size_int32_t[i] == size_int64_t[i]));
  }

  return true;
}

} // namespace

/*
 * Fill the out tensor with random numbers drawn from a normal distribution
 *
 * randn.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
 */
Tensor& randn_out(RuntimeContext& ctx, IntArrayRef size, Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, size) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_KERNEL_CHECK(ctx, check_sizes(size, out.sizes()), InvalidArgument, out);

  // Random number generation setup
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0, 1.0);

  void* out_data = out.mutable_data_ptr();
  if (out_data != nullptr) {
    float* out_data_float = static_cast<float*>(out_data);
    const size_t num_elements = out.numel();

    for (size_t i = 0; i < num_elements; ++i) {
      out_data_float[i] = dist(gen);
    }
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
