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

#include <vector>
#include "paddle/pir/include/dialect/pexpr/adt.h"
#include "paddle/pir/include/dialect/pexpr/data_type.h"
#include "paddle/pir/include/dialect/pexpr/error.h"
#include "paddle/pir/include/dialect/pexpr/type.h"

namespace pexpr {

template <typename T>
struct GetConstVectorPtrNameHelper;

#define SPECIALIZE_GetConstVectorPtrNameHelper(cpp_type, enum_type)            \
  template <>                                                                  \
  struct GetConstVectorPtrNameHelper<cpp_type> {                               \
    static const char* Call() { return "const_std_vector_" #cpp_type "_ptr"; } \
  };
PD_FOR_EACH_DATA_TYPE(SPECIALIZE_GetConstVectorPtrNameHelper);
#undef SPECIALIZE_GetConstVectorPtrNameHelper
template <>
struct GetConstVectorPtrNameHelper<std::string> {
  static const char* Call() { return "const_std_vector_str_ptr"; }
};

template <typename T>
struct TypeImpl<const std::vector<T>*> : public std::monostate {
  using value_type = const std::vector<T>*;

  const char* Name() const { return GetConstVectorPtrNameHelper<T>::Call(); }
};

}  // namespace pexpr
