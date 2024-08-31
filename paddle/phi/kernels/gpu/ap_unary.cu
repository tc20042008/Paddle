// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <mutex>
#include <unordered_map>
#include "glog/logging.h"
#include "jitify.hpp"  // NOLINT
#include "paddle/common/enforce.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/impl/activation_grad_impl.h"
#include "paddle/phi/kernels/impl/activation_impl.h"

#include "paddle/cinn/backends/nvrtc/nvrtc_util.h"
#include "paddle/cinn/runtime/cuda/cuda_module.h"
#include "paddle/phi/kernels/gpu/ap_cuda_jit_util.h"

namespace ap {

template <typename T, typename Context>
void ApUnaryKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   phi::DenseTensor* out) {
  auto generate_ptx = [] {
    ap::Compiler compiler;

    std::string source_code = R"(
  #include <cstdint>
  #define CINN_WITH_CUDA

  extern "C" __global__
  void relu(const float* input, const int num, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < num) {
      output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
  }
  )";
    LOG(ERROR) << "\n" << source_code;
    auto ptx = compiler(source_code);
    CHECK(!ptx.empty());
    return ptx;
  };

  auto ptx = generate_ptx();

  ap::CUDAModule cuda_module(ptx, ap::CUDAModule::Kind::PTX);
  int size = x.numel();
  dim3 blocks_per_grid(1);
  dim3 threads_per_block(100);
  const void* x_data = x.data();
  void* out_data = out->data();
  void* args[] = {&x_data, &size, &out_data};
  cuda_module.LaunchKernel(0, "relu", blocks_per_grid, threads_per_block, args);
}

}  // namespace ap

namespace phi {

template <typename T, typename Context>
void ApUnaryKernel(const Context& dev_ctx,
                   const std::vector<const DenseTensor*>& xs,
                   int num_outputs,
                   const std::string& kernel_definer_lambda,
                   const std::string& define_ctx_maker_lambda,
                   const std::string& kernel_dispatcher_lambda,
                   const std::string& dispatch_ctx_maker_lambda,
                   std::vector<DenseTensor*> outs) {
  PADDLE_ENFORCE_GT(
      xs.size(),
      0,
      phi::errors::InvalidArgument(
          "At least 1 input is required. current number out uts: // %d",
          xs.size()));
  PADDLE_ENFORCE_GT(
      outs.size(),
      0,
      phi::errors::InvalidArgument(
          "num_outputs must be greater than 1. current _outputs: // %d",
          outs.size()));
  for (auto* out : outs) {
    dev_ctx.template Alloc<T>(out);
  }
  ap::ApUnaryKernel<T, Context>(dev_ctx, *xs[0], outs[0]);
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(ap_unary,
                   GPU,
                   ALL_LAYOUT,
                   phi::ApUnaryKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(ap_unary,
                   GPU,
                   ALL_LAYOUT,
                   phi::ApUnaryKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif
