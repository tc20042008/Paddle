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

#include "ap/kernel_dispatch/ap_unary_kernel.h"

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
  const auto& ret =
      ap::kernel_dispatch::ApUnaryKernel(xs,
                                         num_outputs,
                                         kernel_definer_lambda,
                                         define_ctx_maker_lambda,
                                         kernel_dispatcher_lambda,
                                         dispatch_ctx_maker_lambda,
                                         outs);
  PADDLE_ENFORCE(!ret.HasError(),
                 "ap_kernel failed. error_type: %s, error_msg: %s. ",
                 ret.GetError().class_name(),
                 ret.GetError().msg());
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
