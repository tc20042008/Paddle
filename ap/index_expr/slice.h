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

#include <vector>
#include "ap/axpr/adt.h"
#include "ap/axpr/type.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace ap::index_expr {

struct SliceImpl {
  symbol::DimExpr start;
  symbol::DimExpr stop;
  symbol::DimExpr step;

  bool operator==(const SliceImpl& other) const {
    return (other.start == this->start) && (other.stop == this->stop) &&
           (other.step == this->step);
  }
};

DEFINE_ADT_RC(Slice, const SliceImpl);

}  // namespace ap::index_expr

namespace ap::axpr {

template <>
struct TypeImpl<index_expr::Slice> : public std::monostate {
  using value_type = index_expr::Slice;

  const char* Name() const { return "Slice"; }
};

}  // namespace ap::axpr
