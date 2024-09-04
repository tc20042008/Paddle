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
struct GetPointerTypeNameHelper;

#define SPECIALIZE_GET_CPP_TYPE_NAME(cpp_type, enum_type)        \
  template <>                                                    \
  struct GetPointerTypeNameHelper<cpp_type*> {                   \
    static const char* Call() { return #cpp_type "*"; }          \
  };                                                             \
  template <>                                                    \
  struct GetPointerTypeNameHelper<const cpp_type*> {             \
    static const char* Call() { return "const " #cpp_type "*"; } \
  };
PD_FOR_EACH_DATA_TYPE(SPECIALIZE_GET_CPP_TYPE_NAME);
#undef SPECIALIZE_GET_CPP_TYPE_NAME

template <>
struct GetPointerTypeNameHelper<void*> {
  static const char* Call() { return "void*"; }
};

template <>
struct GetPointerTypeNameHelper<const void*> {
  static const char* Call() { return "const void*"; }
};

template <typename T>
struct CppPointerType : public std::monostate {
  using std::monostate::monostate;
  using type = T;
  const char* name() const { return GetPointerTypeNameHelper<T>::Call(); }
};

// clang-format off
using PointerTypeImpl = std::variant<
#define MAKE_POINTER_TYPE_ALTERNATIVE(cpp_type, enum_type)    \
    CppPointerType<cpp_type*>,                                \
    CppPointerType<const cpp_type*>,
    PD_FOR_EACH_DATA_TYPE(MAKE_POINTER_TYPE_ALTERNATIVE)
#undef MAKE_POINTER_TYPE_ALTERNATIVE
    CppPointerType<void*>,
    CppPointerType<const void*>>;
// clang-format on

struct PointerType : public PointerTypeImpl {
  using PointerTypeImpl::PointerTypeImpl;
  DEFINE_ADT_VARIANT_METHODS(PointerTypeImpl);

  const char* name() const {
    return Match([](const auto& impl) { return impl.name(); });
  }
};

}  // namespace pexpr
