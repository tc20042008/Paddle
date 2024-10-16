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

#include "ap/axpr/constants.h"
#include "ap/axpr/data_value.h"
#include "ap/axpr/data_value_util.h"
#include "ap/axpr/method.h"
#include "ap/axpr/method_class.h"

namespace ap::axpr {

namespace detail {

template <typename Val>
Result<Val> ArgValueStaticCast(const Val& self, const std::vector<Val>& args) {
  if (args.size() != 1) {
    return TypeError{std::string() + "'DataValue.cast' take 1 arguments. but " +
                     std::to_string(args.size()) + " were given."};
  }
  const Result<DataValue>& arg_value =
      MethodClass<Val>::template TryGet<DataValue>(self);
  ADT_RETURN_IF_ERR(arg_value);
  const Result<DataType>& arg_type =
      MethodClass<Val>::template TryGet<DataType>(args.at(0));
  ADT_RETURN_IF_ERR(arg_type);
  const auto& data_value =
      arg_value.GetOkValue().StaticCastTo(arg_type.GetOkValue());
  ADT_RETURN_IF_ERR(data_value);
  return data_value.GetOkValue();
}

template <typename ValueT>
adt::Result<ValueT> DataValueGetAttr(const DataValue& data_val,
                                     const std::string& attr_name) {
  if (attr_name == "cast") {
    return ap::axpr::Method<ValueT>{data_val, &ArgValueStaticCast<ValueT>};
  }
  return adt::errors::AttributeError{"'DataValue' object has no attribute '" +
                                     attr_name + "'"};
}

}  // namespace detail

template <typename ValueT>
struct DataValueMethodClass {
  using This = DataValueMethodClass;
  using Self = DataValue;

  adt::Result<ValueT> ToString(const Self& self) {
    ADT_LET_CONST_REF(str, self.ToString());
    return str;
  }

  template <typename BuiltinUnarySymbol>
  static std::optional<BuiltinUnaryFuncT<ValueT>> GetBuiltinUnaryFunc() {
    if constexpr (ConvertBuiltinSymbolToArithmetic<
                      BuiltinUnarySymbol>::convertable) {
      using ArithmeticOp = typename ConvertBuiltinSymbolToArithmetic<
          BuiltinUnarySymbol>::arithmetic_op_type;
      return &This::UnaryFunc<ArithmeticOp>;
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
      return &This::template BinaryFunc<ArithmeticOp>;
    } else if constexpr (std::is_same_v<BultinBinarySymbol,  // NOLINT
                                        builtin_symbol::GetAttr>) {
      return &This::GetAttr;
    } else {
      return std::nullopt;
    }
  }

  static adt::Result<ValueT> GetAttr(const ValueT& obj_val,
                                     const ValueT& attr_name_val) {
    const auto& opt_obj =
        MethodClass<ValueT>::template TryGet<DataValue>(obj_val);
    ADT_RETURN_IF_ERR(opt_obj);
    const auto& obj = opt_obj.GetOkValue();
    const auto& opt_attr_name =
        MethodClass<ValueT>::template TryGet<std::string>(attr_name_val);
    ADT_RETURN_IF_ERR(opt_attr_name);
    const auto& attr_name = opt_attr_name.GetOkValue();
    return detail::DataValueGetAttr<ValueT>(obj, attr_name);
  }

  template <typename ArithmeticOp>
  static adt::Result<ValueT> BinaryFunc(const ValueT& lhs_val,
                                        const ValueT& rhs_val) {
    const auto& opt_lhs =
        MethodClass<ValueT>::template TryGet<DataValue>(lhs_val);
    ADT_RETURN_IF_ERR(opt_lhs);
    const auto& lhs = opt_lhs.GetOkValue();
    const auto& opt_rhs =
        MethodClass<ValueT>::template TryGet<DataValue>(rhs_val);
    ADT_RETURN_IF_ERR(opt_rhs);
    const auto& rhs = opt_rhs.GetOkValue();
    const auto& ret = ArithmeticBinaryFunc<ArithmeticOp>(lhs, rhs);
    ADT_RETURN_IF_ERR(ret);
    return ret.GetOkValue();
  }

  template <typename ArithmeticOp>
  static adt::Result<ValueT> UnaryFunc(const ValueT& val) {
    const auto& opt_operand =
        MethodClass<ValueT>::template TryGet<DataValue>(val);
    ADT_RETURN_IF_ERR(opt_operand);
    const auto& operand = opt_operand.GetOkValue();
    const auto& ret = ArithmeticUnaryFunc<ArithmeticOp>(operand);
    ADT_RETURN_IF_ERR(ret);
    return ret.GetOkValue();
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, DataValue>
    : public DataValueMethodClass<ValueT> {};

namespace detail {

template <typename ValueT>
adt::Result<ValueT> ConstructDataValue(const ValueT&,
                                       const std::vector<ValueT>& args) {
  if (args.size() != 1) {
    return adt::errors::TypeError{
        std::string() + "constructor of 'DataValue' takes 1 arguments, but " +
        std::to_string(args.size()) + " were given."};
  }
  return args.at(0).Match(
      [](bool c) -> adt::Result<ValueT> { return DataValue{c}; },
      [](int64_t c) -> adt::Result<ValueT> { return DataValue{c}; },
      [](const DataValue& c) -> adt::Result<ValueT> { return c; },
      [](const auto& impl) -> adt::Result<ValueT> {
        using T = std::decay_t<decltype(impl)>;
        return adt::errors::TypeError{
            std::string() +
            "unsupported operand type for constructor of 'DataValue': '" +
            TypeImpl<T>{}.Name() + "'"};
      });
}

}  // namespace detail

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<DataValue>> {
  template <typename BuiltinUnarySymbol>
  static std::optional<BuiltinUnaryFuncT<ValueT>> GetBuiltinUnaryFunc() {
    if constexpr (std::is_same_v<BuiltinUnarySymbol, builtin_symbol::Call>) {
      return &UnaryFuncReturnCapturedValue<ValueT,
                                           &detail::ConstructDataValue<ValueT>>;
    } else {
      return std::nullopt;
    }
  }

  template <typename BultinBinarySymbol>
  static std::optional<BuiltinBinaryFuncT<ValueT>> GetBuiltinBinaryFunc() {
    return std::nullopt;
  }
};

}  // namespace ap::axpr
