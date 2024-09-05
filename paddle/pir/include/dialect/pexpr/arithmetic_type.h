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
#include "paddle/pir/include/dialect/pexpr/data_type.h"

namespace pexpr {

template <typename T>
struct GetArithmeticTypeNameHelper;

#define SPECIALIZE_GET_CPP_TYPE_NAME(cpp_type, enum_type) \
  template <>                                             \
  struct GetArithmeticTypeNameHelper<cpp_type> {          \
    static const char* Call() { return #cpp_type; }       \
  };
PD_FOR_EACH_DATA_TYPE(SPECIALIZE_GET_CPP_TYPE_NAME);
#undef SPECIALIZE_GET_CPP_TYPE_NAME
template <>
struct GetArithmeticTypeNameHelper<adt::Undefined> {
  static const char* Call() { return "undefined"; }
};

template <typename T>
struct CppArithmeticType : public std::monostate {
  using std::monostate::monostate;
  using type = T;
  const char* Name() const { return GetArithmeticTypeNameHelper<T>::Call(); }
};

// clang-format off
using ArithmeticTypeImpl = std::variant<
#define MAKE_ARITHMETIC_TYPE_ALTERNATIVE(cpp_type, enum_type)    \
    CppArithmeticType<cpp_type>,
    PD_FOR_EACH_DATA_TYPE(MAKE_ARITHMETIC_TYPE_ALTERNATIVE)
#undef MAKE_ARITHMETIC_TYPE_ALTERNATIVE
    CppArithmeticType<adt::Undefined>>;
// clang-format on

struct ArithmeticType : public ArithmeticTypeImpl {
  using ArithmeticTypeImpl::ArithmeticTypeImpl;
  DEFINE_ADT_VARIANT_METHODS(ArithmeticTypeImpl);

  const char* Name() const {
    return Match([](const auto& impl) { return impl.Name(); });
  }
};

}  // namespace pexpr
