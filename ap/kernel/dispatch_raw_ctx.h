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

#pragma once
#include "ap/axpr/type.h"
#include "ap/kernel/adt.h"
#include "ap/kernel/arg_value.h"
#include "ap/kernel/data_type.h"
#include "ap/kernel/typed_buffer.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

class DenseTensor;

}

namespace ap::kernel_dispatch {

class CudaModule {
 public:
  virtual ~CudaModule() = default;

  virtual adt::Result<adt::Ok> LaunchCudaKernel(
      const std::string& func_name,
      int64_t num_blocks,
      int64_t num_threads,
      const std::vector<void*>& args) = 0;

 protected:
  CudaModule() = default;
};

template <typename ValueT>
struct DispatchRawCtxImpl {
  adt::List<ValueT> inputs;
  adt::List<ValueT> outputs;
  std::shared_ptr<CudaModule> cuda_module;
  std::unordered_map<std::string, adt::List<kernel_define::ArgType>>
      func_name2arg_types;

  bool operator==(const DispatchRawCtxImpl& other) const {
    return &other == this;
  }

  Result<adt::Ok> LaunchCudaKernel(
      const std::string& func_name,
      int64_t num_blocks,
      int64_t num_threads,
      const adt::List<ArgValue>& kernel_args) const;
};

template <typename ValueT>
DEFINE_ADT_RC(DispatchRawCtx, DispatchRawCtxImpl<ValueT>);

}  // namespace ap::kernel_dispatch

namespace ap::axpr {

template <typename ValueT>
struct TypeImpl<ap::kernel_dispatch::DispatchRawCtx<ValueT>>
    : public std::monostate {
  using value_type = ap::kernel_dispatch::DispatchRawCtx<ValueT>;

  const char* Name() const { return "DispatchRawCtx"; }
};

}  // namespace ap::axpr
