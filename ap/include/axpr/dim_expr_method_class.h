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

#include "ap/include/axpr/method_class.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace ap::axpr {

template <typename ValueT>
struct DimExprMethodClass {
  using This = DimExprMethodClass;
  using Self = symbol::DimExpr;

  static adt::Result<ValueT> ToString(const ValueT& self_val,
                                      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return symbol::ToString(self);
  }

  static adt::Result<ValueT> Hash(const ValueT& self_val,
                                  const std::vector<ValueT>&) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    int64_t hash_value = std::hash<Self>()(self);
    return hash_value;
  }
};

template <typename ValueT>
const axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>& GetDimExprClass() {
  using ClassT = axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>;
  static ClassT cls(
      axpr::MakeBuiltinClass<ValueT>("DimExpr", [&](const auto& Define) {
        Define("__str__", &DimExprMethodClass<ValueT>::ToString);
        Define("__hash__", &DimExprMethodClass<ValueT>::Hash);
      }));
  return cls;
}

}  // namespace ap::axpr
