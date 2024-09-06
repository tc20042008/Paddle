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

#include "paddle/pir/include/dialect/pexpr/arithmetic_value.h"
#include "paddle/pir/include/dialect/pexpr/arithmetic_value_util.h"
#include "paddle/pir/include/dialect/pexpr/constants.h"
#include "paddle/pir/include/dialect/pexpr/method_class.h"

namespace pexpr {

template <typename ValueT>
struct ArithmeticValueMethodClass {
  using Self = ArithmeticValueMethodClass;

  static const char* Name() { return "ArithmeticValue"; }

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
    const auto& opt_lhs =
        MethodClass<ValueT>::template TryGet<ArithmeticValue>(lhs_val);
    ADT_RETURN_IF_ERROR(opt_lhs);
    const auto& lhs = opt_lhs.GetOkValue();
    const auto& opt_rhs =
        MethodClass<ValueT>::template TryGet<ArithmeticValue>(rhs_val);
    ADT_RETURN_IF_ERROR(opt_rhs);
    const auto& rhs = opt_rhs.GetOkValue();
    const auto& ret = ArithmeticBinaryFunc<ArithmeticOp>(lhs, rhs);
    ADT_RETURN_IF_ERROR(ret);
    return ret.GetOkValue();
  }

  template <typename ArithmeticOp>
  static adt::Result<ValueT> UnaryFunc(const ValueT& val) {
    const auto& opt_operand =
        MethodClass<ValueT>::template TryGet<ArithmeticValue>(val);
    ADT_RETURN_IF_ERROR(opt_operand);
    const auto& operand = opt_operand.GetOkValue();
    const auto& ret = ArithmeticUnaryFunc<ArithmeticOp>(operand);
    ADT_RETURN_IF_ERROR(ret);
    return ret.GetOkValue();
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, ArithmeticValue> {
  using method_class = ArithmeticValueMethodClass<ValueT>;

  static const char* Name() { return method_class::Name(); }

  template <typename BuiltinUnarySymbol>
  static std::optional<BuiltinUnaryFuncT<ValueT>> GetBuiltinUnaryFunc() {
    return method_class::template GetBuiltinUnaryFunc<BuiltinUnarySymbol>();
  }

  template <typename BultinBinarySymbol>
  static std::optional<BuiltinBinaryFuncT<ValueT>> GetBuiltinBinaryFunc() {
    return method_class::template GetBuiltinBinaryFunc<BultinBinarySymbol>();
  }
};

}  // namespace pexpr
