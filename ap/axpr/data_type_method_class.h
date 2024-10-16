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
#include "ap/axpr/data_type.h"
#include "ap/axpr/int_data_type.h"
#include "ap/axpr/method_class.h"

namespace ap::axpr {

template <typename ValueT>
struct DataTypeMethodClass {
  using This = DataTypeMethodClass;
  using Self = DataType;

  adt::Result<ValueT> ToString(const Self& data_type) {
    return std::string("DataType.") + data_type.Name();
  }

  template <typename BuiltinUnarySymbol>
  static std::optional<BuiltinUnaryFuncT<ValueT>> GetBuiltinUnaryFunc() {
    return std::nullopt;
  }

  template <typename BultinBinarySymbol>
  static std::optional<BuiltinBinaryFuncT<ValueT>> GetBuiltinBinaryFunc() {
    if constexpr (std::is_same_v<BultinBinarySymbol, builtin_symbol::EQ>) {
      return &This::EQ;
    } else if constexpr (std::is_same_v<BultinBinarySymbol,  // NOLINT
                                        builtin_symbol::NE>) {
      return &This::NE;
    } else {
      std::nullopt;
    }
  }

  static Result<ValueT> EQ(const ValueT& lhs_val, const ValueT& rhs_val) {
    const auto& opt_lhs =
        MethodClass<ValueT>::template TryGet<DataType>(lhs_val);
    ADT_RETURN_IF_ERR(opt_lhs);
    const auto& lhs = opt_lhs.GetOkValue();
    const auto& opt_rhs =
        MethodClass<ValueT>::template TryGet<DataType>(rhs_val);
    ADT_RETURN_IF_ERR(opt_rhs);
    const auto& rhs = opt_rhs.GetOkValue();
    const auto& pattern_match =
        ::common::Overloaded{[](auto lhs, auto rhs) -> ValueT {
          return std::is_same_v<decltype(lhs), decltype(rhs)>;
        }};
    return std::visit(pattern_match, lhs.variant(), rhs.variant());
  }

  static Result<ValueT> NE(const ValueT& lhs_val, const ValueT& rhs_val) {
    const auto& opt_lhs =
        MethodClass<ValueT>::template TryGet<DataType>(lhs_val);
    ADT_RETURN_IF_ERR(opt_lhs);
    const auto& lhs = opt_lhs.GetOkValue();
    const auto& opt_rhs =
        MethodClass<ValueT>::template TryGet<DataType>(rhs_val);
    ADT_RETURN_IF_ERR(opt_rhs);
    const auto& rhs = opt_rhs.GetOkValue();
    const auto& pattern_match =
        ::common::Overloaded{[](auto lhs, auto rhs) -> ValueT {
          return !std::is_same_v<decltype(lhs), decltype(rhs)>;
        }};
    return std::visit(pattern_match, lhs.variant(), rhs.variant());
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, DataType> : public DataTypeMethodClass<ValueT> {
};

template <typename ValueT>
struct TypeImplDataTypeMethodClass {
  using This = TypeImplDataTypeMethodClass;
  using Self = TypeImpl<DataType>;

  adt::Result<ValueT> GetAttr(const Self&, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, TryGetImpl<std::string>(attr_name_val));
    static const std::unordered_map<std::string, DataType> map{
#define MAKE_CPP_TYPE_CASE(cpp_type, enum_type)                              \
  {axpr::CppDataType<cpp_type>{}.Name(), DataType{CppDataType<cpp_type>{}}}, \
      {axpr::CppDataType<const cpp_type>{}.Name(),                           \
       DataType{                                                             \
           CppDataType<cpp_type>{}}},  // it's not a typo, DataType.const_int8
                                       // and DataType.int8 are treated
                                       // identical.

        PD_FOR_EACH_DATA_TYPE(MAKE_CPP_TYPE_CASE)
#undef MAKE_CPP_TYPE_CASE

#define MAKE_INT_CPP_TYPE_CASE(cpp_type)              \
  {#cpp_type, DataType{CppDataType<cpp_type##_t>{}}}, \
      {"const_" #cpp_type, DataType{CppDataType<cpp_type##_t>{}}},

            AP_FOR_EACH_INT_TYPE(MAKE_INT_CPP_TYPE_CASE)
#undef MAKE_INT_CPP_TYPE_CASE
    };
    const auto iter = map.find(attr_name);
    if (iter != map.end()) {
      return iter->second;
    }
    return adt::errors::AttributeError{
        std::string() + "class 'DataType' has no static attribute '" +
        attr_name + "'."};
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<DataType>>
    : public TypeImplDataTypeMethodClass<ValueT> {};

}  // namespace ap::axpr
