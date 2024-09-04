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
#include "paddle/pir/include/dialect/pexpr/arithmetic_type.h"
#include "paddle/pir/include/dialect/pexpr/data_type.h"

namespace pexpr {

using ArithmeticValueImpl = std::variant<
#define MAKE_ARG_VALUE_ALTERNATIVE(cpp_type, enum_type) cpp_type,
    PD_FOR_EACH_DATA_TYPE(MAKE_ARG_VALUE_ALTERNATIVE) adt::Undefined
#undef MAKE_ARG_VALUE_ALTERNATIVE
    >;

struct ArithmeticValue : public ArithmeticValueImpl {
  using ArithmeticValueImpl::ArithmeticValueImpl;
  DEFINE_ADT_VARIANT_METHODS(ArithmeticValueImpl);

  ArithmeticType GetType() const {
    return Match([](auto impl) -> ArithmeticType {
      return CppArithmeticType<decltype(impl)>{};
    });
  }

  template <typename T>
  Result<T> TryGet() const {
    if (!Has<T>()) {
      return adt::errors::TypeError{
          std::string() + "ArithmeticValue::TryGet() failed. expected_type: " +
          CppArithmeticType<T>{}.name() + ", actual_type: " + GetType().name()};
    }
    return Get<T>();
  }

  Result<ArithmeticValue> StaticCastTo(const ArithmeticType& dst_type) const {
    const auto& pattern_match = ::common::Overloaded{
        [&](auto arg_type_impl,
            auto cpp_value_impl) -> Result<ArithmeticValue> {
          using DstT = typename decltype(arg_type_impl)::type;
          return ArithmeticValueStaticCast<DstT>(cpp_value_impl);
        }};
    return std::visit(pattern_match, dst_type.variant(), this->variant());
  }

 private:
  template <typename DstT, typename SrcT>
  Result<ArithmeticValue> ArithmeticValueStaticCast(SrcT v) const {
    if constexpr (std::is_same_v<DstT, adt::Undefined>) {
      return adt::errors::TypeError{
          "static_cast can not cast to 'undefined' type."};
    } else if constexpr (std::is_same_v<DstT, phi::dtype::pstring>) {
      return adt::errors::TypeError{
          "static_cast can not cast to 'pstring' type."};
    } else if constexpr (std::is_same_v<SrcT, adt::Undefined>) {
      return adt::errors::TypeError{
          "static_cast can not cast from 'undefined' type."};
    } else if constexpr (std::is_same_v<SrcT, phi::dtype::pstring>) {
      return adt::errors::TypeError{
          "static_cast can not cast from 'pstring' type."};
    } else {
      return static_cast<DstT>(v);
    }
  }
};

}  // namespace pexpr
