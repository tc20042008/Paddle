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

#include "paddle/pir/include/dialect/pexpr/pointer_type.h"

namespace pexpr {

namespace detail {

template <typename T>
struct TypeConverter;

#define SPECIALIZE_TYPE_CONVERTER(cpp_type, enum_type)    \
  template <>                                             \
  struct TypeConverter<CppPointerType<cpp_type*>> {       \
    using remove_const_type = CppPointerType<cpp_type*>;  \
  };                                                      \
  template <>                                             \
  struct TypeConverter<CppPointerType<const cpp_type*>> { \
    using remove_const_type = CppPointerType<cpp_type*>;  \
  };

PD_FOR_EACH_DATA_TYPE(SPECIALIZE_TYPE_CONVERTER);
#undef SPECIALIZE_TYPE_CONVERTER

template <>
struct TypeConverter<CppPointerType<void*>> {
  using remove_const_type = CppPointerType<void*>;
};

template <>
struct TypeConverter<CppPointerType<const void*>> {
  using remove_const_type = CppPointerType<void*>;
};

}  // namespace detail

PointerType RemoveConst(const PointerType& ptr_type) const {
  return ptr_type.Match([](auto impl) {
    return PointerType{
        typename detail::TypeConverter<decltype(impl)>::remove_const_type{}};
  });
}

}  // namespace pexpr
