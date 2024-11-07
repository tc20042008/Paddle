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
#include "ap/axpr/type.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace ap::kernel_define {

template <typename IrNodeT>
struct DimExprKernelArgIdImpl {
  symbol::DimExpr dim_expr;

  bool operator==(const DimExprKernelArgIdImpl& other) const {
    return this->dim_expr == other.dim_expr;
  }

  template <typename ValueT>
  adt::Result<ValueT> CastData() const {
    return ValueT{this->dim_expr};
  }

  std::size_t GetHashValue() const {
    return std::hash<symbol::DimExpr>()(this->dim_expr);
  }
};

template <typename IrNodeT>
DEFINE_ADT_RC(DimExprKernelArgId, DimExprKernelArgIdImpl<IrNodeT>);

}  // namespace ap::kernel_define

namespace ap::axpr {

template <typename IrNodeT>
struct TypeImpl<kernel_define::DimExprKernelArgId<IrNodeT>>
    : public std::monostate {
  using std::monostate::monostate;

  const char* Name() const { return "DimExprKernelArgId"; }
};

}  // namespace ap::axpr
