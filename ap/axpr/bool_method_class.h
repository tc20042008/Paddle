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

#include <string>
#include "ap/axpr/bool_int_double_arithmetic_util.h"
#include "ap/axpr/constants.h"
#include "ap/axpr/method_class.h"

namespace ap::axpr {

template <typename ValueT>
struct BoolMethodClass {
  using Self = BoolMethodClass;

  template <typename BuiltinUnarySymbol>
  static std::optional<BuiltinUnaryFuncT<ValueT>> GetBuiltinUnaryFunc() {
    if constexpr (ConvertBuiltinSymbolToArithmetic<
                      BuiltinUnarySymbol>::convertable) {
      using ArithmeticOp = typename ConvertBuiltinSymbolToArithmetic<
          BuiltinUnarySymbol>::arithmetic_op_type;
      return &Self::UnaryFunc<ArithmeticOp>;
    } else {
      return std::nullopt;
    }
  }

  template <typename BultinBinarySymbol>
  static std::optional<BuiltinBinaryFuncT<ValueT>> GetBuiltinBinaryFunc() {
    if constexpr (ConvertBuiltinSymbolToArithmetic<
                      BultinBinarySymbol>::convertable) {
      using ArithmeticOp = typename ConvertBuiltinSymbolToArithmetic<
          BultinBinarySymbol>::arithmetic_op_type;
      return &Self::template BinaryFunc<ArithmeticOp>;
    } else {
      return std::nullopt;
    }
  }

  template <typename ArithmeticOp>
  static adt::Result<ValueT> BinaryFunc(const ValueT& lhs_val,
                                        const ValueT& rhs_val) {
    const auto& opt_lhs = MethodClass<ValueT>::template TryGet<bool>(lhs_val);
    ADT_RETURN_IF_ERROR(opt_lhs);
    bool lhs = opt_lhs.GetOkValue();
    return rhs_val.Match(
        [&](bool rhs) -> adt::Result<ValueT> {
          return BoolIntDoubleArithmeticBinaryFunc<ArithmeticOp, ValueT>(lhs,
                                                                         rhs);
        },
        [&](int64_t rhs) -> adt::Result<ValueT> {
          return BoolIntDoubleArithmeticBinaryFunc<ArithmeticOp, ValueT>(lhs,
                                                                         rhs);
        },
        [&](double rhs) -> adt::Result<ValueT> {
          return BoolIntDoubleArithmeticBinaryFunc<ArithmeticOp, ValueT>(lhs,
                                                                         rhs);
        },
        [&](const auto& impl) -> adt::Result<ValueT> {
          using T = std::decay_t<decltype(impl)>;
          return adt::errors::TypeError{
              std::string() + "unsupported operand type(s) for " +
              ArithmeticOp::Name() + ": 'bool' and '" + TypeImpl<T>{}.Name() +
              "'"};
        });
  }

  template <typename ArithmeticOp>
  static adt::Result<ValueT> UnaryFunc(const ValueT& val) {
    const auto& opt_operand = MethodClass<ValueT>::template TryGet<bool>(val);
    ADT_RETURN_IF_ERROR(opt_operand);
    bool operand = opt_operand.GetOkValue();
    return BoolIntDoubleArithmeticUnaryFunc<ArithmeticOp, ValueT>(operand);
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, bool> {
  using method_class = BoolMethodClass<ValueT>;

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
struct MethodClassImpl<ValueT, TypeImpl<bool>>
    : public EmptyMethodClass<ValueT> {};

}  // namespace ap::axpr
