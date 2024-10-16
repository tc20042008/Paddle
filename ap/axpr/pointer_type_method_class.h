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
#include "ap/axpr/int_data_type.h"
#include "ap/axpr/method_class.h"
#include "ap/axpr/pointer_type.h"

namespace ap::axpr {

template <typename ValueT>
struct PointerTypeMethodClass {
  using This = PointerTypeMethodClass;
  using Self = PointerType;

  adt::Result<ValueT> ToString(const Self& self) {
    return std::string("PointerType.") + self.Name();
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
        MethodClass<ValueT>::template TryGet<PointerType>(lhs_val);
    ADT_RETURN_IF_ERR(opt_lhs);
    const auto& lhs = opt_lhs.GetOkValue();
    const auto& opt_rhs =
        MethodClass<ValueT>::template TryGet<PointerType>(rhs_val);
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
        MethodClass<ValueT>::template TryGet<PointerType>(lhs_val);
    ADT_RETURN_IF_ERR(opt_lhs);
    const auto& lhs = opt_lhs.GetOkValue();
    const auto& opt_rhs =
        MethodClass<ValueT>::template TryGet<PointerType>(rhs_val);
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
struct MethodClassImpl<ValueT, PointerType>
    : public PointerTypeMethodClass<ValueT> {};

template <typename ValueT>
struct TypeImplPointerTypeMethodClass {
  using This = TypeImplPointerTypeMethodClass;
  using Self = TypeImpl<PointerType>;

  template <typename T>
  const char* PtrTypeName() {
    return axpr::CppPointerType<T>{}.Name();
  }

  template <typename T>
  PointerType PtrType() {
    return PointerType{CppPointerType<T>{}};
  }

  adt::Result<ValueT> GetAttr(const Self&, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, TryGetImpl<std::string>(attr_name_val));
    static const std::unordered_map<std::string, PointerType> map{
#define MAKE_CPP_TYPE_CASE(cpp_type, enum_type)     \
  {PtrTypeName<cpp_type*>(), PtrType<cpp_type*>()}, \
      {PtrTypeName<const cpp_type*>(), PtrType<const cpp_type*>()},
        PD_FOR_EACH_DATA_TYPE(MAKE_CPP_TYPE_CASE)
#undef MAKE_CPP_TYPE_CASE
#define MAKE_INT_CPP_TYPE_CASE(cpp_type)        \
  {#cpp_type "_ptr", PtrType<cpp_type##_t*>()}, \
      {"const_" #cpp_type "_ptr", PtrType<const cpp_type##_t*>()},
            AP_FOR_EACH_INT_TYPE(MAKE_INT_CPP_TYPE_CASE)
#undef MAKE_INT_CPP_TYPE_CASE
                {PtrTypeName<void*>(), PtrType<void*>()},
        {PtrTypeName<const void*>(), PtrType<const void*>()},
    };
    const auto iter = map.find(attr_name);
    if (iter != map.end()) {
      return iter->second;
    }
    return adt::errors::AttributeError{
        std::string() + "class 'PointerType' has no static attribute '" +
        attr_name + "'."};
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<PointerType>>
    : public TypeImplPointerTypeMethodClass<ValueT> {};

}  // namespace ap::axpr
