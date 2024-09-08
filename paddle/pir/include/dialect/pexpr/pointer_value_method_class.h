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

#include "paddle/pir/include/dialect/pexpr/constants.h"
#include "paddle/pir/include/dialect/pexpr/method_class.h"
#include "paddle/pir/include/dialect/pexpr/pointer_value.h"

namespace pexpr {

template <typename ValueT>
struct PointerValueMethodClass {
  using Self = PointerValueMethodClass;

  template <typename BuiltinUnarySymbol>
  static std::optional<BuiltinUnaryFuncT<ValueT>> GetBuiltinUnaryFunc() {
    return std::nullopt;
  }

  template <typename BultinBinarySymbol>
  static std::optional<BuiltinBinaryFuncT<ValueT>> GetBuiltinBinaryFunc() {
    if constexpr (std::is_same_v<BultinBinarySymbol, builtin_symbol::EQ>) {
      return &Self::EQ;
    } else if constexpr (std::is_same_v<BultinBinarySymbol,  // NOLINT
                                        builtin_symbol::NE>) {
      return &Self::NE;
    } else {
      std::nullopt;
    }
  }

  static Result<ValueT> EQ(const ValueT& lhs_val, const ValueT& rhs_val) {
    const auto& opt_lhs =
        MethodClass<ValueT>::template TryGet<PointerValue>(lhs_val);
    ADT_RETURN_IF_ERROR(opt_lhs);
    const auto& lhs = opt_lhs.GetOkValue();
    const auto& opt_rhs =
        MethodClass<ValueT>::template TryGet<PointerValue>(rhs_val);
    ADT_RETURN_IF_ERROR(opt_rhs);
    const auto& rhs = opt_rhs.GetOkValue();
    const auto& pattern_match =
        ::common::Overloaded{[](auto lhs, auto rhs) -> ValueT {
          if constexpr (std::is_same_v<decltype(lhs), decltype(rhs)>) {
            return lhs == rhs;
          } else {
            return false;
          }
        }};
    return std::visit(pattern_match, lhs.variant(), rhs.variant());
  }

  static Result<ValueT> NE(const ValueT& lhs_val, const ValueT& rhs_val) {
    const auto& opt_lhs =
        MethodClass<ValueT>::template TryGet<PointerValue>(lhs_val);
    ADT_RETURN_IF_ERROR(opt_lhs);
    const auto& lhs = opt_lhs.GetOkValue();
    const auto& opt_rhs =
        MethodClass<ValueT>::template TryGet<PointerValue>(rhs_val);
    ADT_RETURN_IF_ERROR(opt_rhs);
    const auto& rhs = opt_rhs.GetOkValue();
    const auto& pattern_match =
        ::common::Overloaded{[](auto lhs, auto rhs) -> ValueT {
          if constexpr (std::is_same_v<decltype(lhs), decltype(rhs)>) {
            return lhs != rhs;
          } else {
            return true;
          }
        }};
    return std::visit(pattern_match, lhs.variant(), rhs.variant());
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, PointerValue> {
  using method_class = PointerValueMethodClass<ValueT>;

  template <typename BuiltinUnarySymbol>
  static std::optional<BuiltinUnaryFuncT<ValueT>> GetBuiltinUnaryFunc() {
    return method_class::template GetBuiltinUnaryFunc<BuiltinUnarySymbol>();
  }

  template <typename BultinBinarySymbol>
  static std::optional<BuiltinBinaryFuncT<ValueT>> GetBuiltinBinaryFunc() {
    return method_class::template GetBuiltinBinaryFunc<BultinBinarySymbol>();
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<PointerValue>>
    : public EmptyMethodClass<ValueT> {};

}  // namespace pexpr
