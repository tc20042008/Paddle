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

#include "paddle/pir/include/dialect/pexpr/arithmetic_type.h"
#include "paddle/pir/include/dialect/pexpr/arithmetic_value.h"
#include "paddle/pir/include/dialect/pexpr/binary_func.h"
#include "paddle/pir/include/dialect/pexpr/unary_func.h"

namespace pexpr {

namespace detail {

template <typename ArithmeticOp>
struct ArithmeticUnaryFuncHelper {
  static Result<ArithmeticValue> Call(const ArithmeticValue& value) {
    return value.Match([](auto val) -> Result<ArithmeticValue> {
      if constexpr (IsArithmeticOpSupported<decltype(val)>()) {
        return ArithmeticOp::Call(val);
      } else {
        return adt::errors::TypeError{
            std::string() + "unsupported operand type for " +
            ArithmeticOp::Name() + ": " +
            CppArithmeticType<decltype(val)>{}.Name() + "."};
      }
    });
  }
};

template <typename ArithmeticOp>
struct ArithmeticBinaryOpHelper {
  template <typename LhsT, typename RhsT>
  static Result<ArithmeticValue> Call(LhsT lhs, RhsT rhs) {
    return ArithmeticOp::Call(lhs, rhs);
  }
};

template <>
struct ArithmeticBinaryOpHelper<ArithmeticDiv> {
  template <typename LhsT, typename RhsT>
  static Result<ArithmeticValue> Call(LhsT lhs, RhsT rhs) {
    if (rhs == 0) {
      return adt::errors::ZeroDivisionError{"division by zero"};
    }
    return ArithmeticDiv::Call(lhs, rhs);
  }
};

template <>
struct ArithmeticBinaryOpHelper<ArithmeticMod> {
  template <typename LhsT, typename RhsT>
  static Result<ArithmeticValue> Call(LhsT lhs, RhsT rhs) {
    if constexpr (std::is_integral_v<LhsT> && std::is_integral_v<RhsT>) {
      return ArithmeticMod::Call(lhs, rhs);
    } else if constexpr (!std::is_integral_v<LhsT>) {
      return adt::errors::TypeError{
          "'%' only support intergral type. 'lhs' is not a intergral type"};
    } else {
      return adt::errors::TypeError{
          "'%' only support intergral type. 'rhs' is not a intergral type"};
    }
  }
};

template <typename ArithmeticOp>
struct ArithmeticBinaryFuncHelper {
  static Result<ArithmeticValue> Call(const ArithmeticValue& lhs_value,
                                      const ArithmeticValue& rhs_value) {
    const auto& pattern_match =
        ::common::Overloaded{[](auto lhs, auto rhs) -> Result<ArithmeticValue> {
          if constexpr (IsArithmeticOpSupported<decltype(lhs)>() &&
                        IsArithmeticOpSupported<decltype(rhs)>()) {
            return ArithmeticBinaryOpHelper<ArithmeticOp>::Call(lhs, rhs);
          } else {
            return adt::errors::TypeError{
                std::string() + "unsupported operand types for " +
                ArithmeticOp::Name() + ": '" +
                CppArithmeticType<decltype(lhs)>{}.Name() + "' and '" +
                CppArithmeticType<decltype(rhs)>{}.Name() + "'."};
          }
        }};
    return std::visit(pattern_match, lhs_value.variant(), rhs_value.variant());
  }
};

}  // namespace detail

template <typename ArithmeticOp>
Result<ArithmeticValue> ArithmeticUnaryFunc(const ArithmeticValue& value) {
  return detail::ArithmeticUnaryFuncHelper<ArithmeticOp>::Call(value);
}

template <typename ArithmeticOp>
Result<ArithmeticValue> ArithmeticBinaryFunc(const ArithmeticValue& lhs_value,
                                             const ArithmeticValue& rhs_value) {
  return detail::ArithmeticBinaryFuncHelper<ArithmeticOp>::Call(lhs_value,
                                                                rhs_value);
}

}  // namespace pexpr
