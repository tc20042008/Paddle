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

#include "paddle/pir/include/dialect/pexpr/adt.h"
#include "paddle/pir/include/dialect/pexpr/binary_func.h"
#include "paddle/pir/include/dialect/pexpr/error.h"
#include "paddle/pir/include/dialect/pexpr/unary_func.h"

namespace pexpr {

template <typename ArithmeticOp, typename ValueT, typename T>
Result<ValueT> BoolIntDoubleArithmeticUnaryFunc(const T& value) {
  return BoolIntDoubleUnary<ArithmeticOp>::Call(value);
}

template <typename ArithmeticOp, typename ValueT, typename T0, typename T1>
Result<ValueT> BoolIntDoubleArithmeticBinaryFunc(const T0& lhs, const T1& rhs) {
  if constexpr (std::is_same_v<ArithmeticOp, builtin_symbol::Div> ||
                std::is_same_v<ArithmeticOp, builtin_symbol::Mod>) {
    if (rhs == 0) {
      return adt::errors::ZeroDivisionError{"division or modulo by zero"};
    }
  }
  return BoolIntDoubleBinary<ArithmeticOp>::Call(lhs, rhs);
}

}  // namespace pexpr
