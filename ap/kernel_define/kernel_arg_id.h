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

#include "ap/adt/adt.h"
#include "ap/kernel_define/dim_expr_kernel_arg_id.h"
#include "ap/kernel_define/in_tensor_data_ptr_kernel_arg_id.h"
#include "ap/kernel_define/out_tensor_data_ptr_kernel_arg_id.h"

namespace ap::kernel_define {

template <typename IrNodeT>
using KernelArgIdImpl = std::variant<DimExprKernelArgId<IrNodeT>,
                                     InTensorDataPtrKernelArgId<IrNodeT>,
                                     OutTensorDataPtrKernelArgId<IrNodeT>>;

template <typename IrNodeT>
struct KernelArgId : public KernelArgIdImpl<IrNodeT> {
  using KernelArgIdImpl<IrNodeT>::KernelArgIdImpl;

  DEFINE_ADT_VARIANT_METHODS(KernelArgIdImpl<IrNodeT>);

  template <typename ValueT>
  ValueT CastTo() const {
    return Match([](const auto& impl) -> ValueT { return impl; });
  }

  template <typename ValueT>
  static adt::Result<KernelArgId> CastFrom(const ValueT& val) {
    using RetT = adt::Result<KernelArgId>;
    return val.Match(
        [](const DimExprKernelArgId<IrNodeT>& impl) -> RetT { return impl; },
        [](const InTensorDataPtrKernelArgId<IrNodeT>& impl) -> RetT {
          return impl;
        },
        [](const OutTensorDataPtrKernelArgId<IrNodeT>& impl) -> RetT {
          return impl;
        },
        [](const auto& impl) -> RetT {
          return adt::errors::TypeError{"KernelArgId::CastFrom() failed."};
        });
  }

  std::size_t GetHashValue() const {
    std::size_t hash_value = Match(
        [&](const auto& impl) -> std::size_t { return impl->GetHashValue(); });
    return adt::hash_combine(this->index(), hash_value);
  }
};

}  // namespace ap::kernel_define

namespace std {

template <typename IrNodeT>
struct hash<ap::kernel_define::KernelArgId<IrNodeT>> {
  std::size_t operator()(
      const ap::kernel_define::KernelArgId<IrNodeT>& arg_id) const {
    return arg_id.GetHashValue();
  }
};

}  // namespace std
