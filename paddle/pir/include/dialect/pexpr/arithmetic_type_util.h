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

namespace pexpr {

inline Result<ArithmeticType> GetArithmeticTypeFromPhiDataType(
    ::phi::DataType data_type) {
  static const std::unordered_map<::phi::DataType, ArithmeticType> map{
#define MAKE_PHI_DATA_TYPE_TO_ARG_TYPE_CASE(cpp_type, enum_type) \
  {::phi::enum_type, ArithmeticType{CppArithmeticType<cpp_type>{}}},
      PD_FOR_EACH_DATA_TYPE(MAKE_PHI_DATA_TYPE_TO_ARG_TYPE_CASE)
#undef MAKE_PHI_DATA_TYPE_TO_ARG_TYPE_CASE
  };
  const auto& iter = map.find(data_type);
  if (iter == map.end()) {
    return InvalidArgumentError{"Invalid phi data type."};
  }
  return iter->second;
}

namespace detail {

template <typename DstT, typename SrcT>
Result<ArithmeticValue> ArithmeticValueStaticCast(SrcT v) {
  if constexpr (std::is_same_v<DstT, adt::Undefined>) {
    return TypeError{"static_cast can not cast to 'undefined' type."};
  } else if constexpr (std::is_same_v<DstT, phi::dtype::pstring>) {
    return TypeError{"static_cast can not cast to 'pstring' type."};
  } else if constexpr (std::is_same_v<SrcT, adt::Undefined>) {
    return TypeError{"static_cast can not cast from 'undefined' type."};
  } else if constexpr (std::is_same_v<SrcT, phi::dtype::pstring>) {
    return TypeError{"static_cast can not cast from 'pstring' type."};
  } else {
    return static_cast<DstT>(v);
  }
}

}  // namespace detail

inline Result<ArithmeticValue> ArithmeticValueStaticCast(
    const ArithmeticType& dst_type, const ArithmeticValue& value) {
  const auto& pattern_match = ::common::Overloaded{
      [&](auto arg_type_impl, auto cpp_value_impl) -> Result<ArithmeticValue> {
        using DstT = typename decltype(arg_type_impl)::type;
        return detail::ArithmeticValueStaticCast<DstT>(cpp_value_impl);
      }};
  return std::visit(pattern_match, dst_type.variant(), value.variant());
}

}  // namespace pexpr
