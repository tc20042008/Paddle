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
#include "ap/axpr/adt.h"
#include "ap/axpr/data_type.h"
#include "ap/axpr/error.h"
#include "ap/axpr/type.h"

namespace ap::axpr {

template <typename T>
struct GetVectorPtrNameHelper;

#define SPECIALIZE_GetVectorPtrNameHelper(cpp_type, enum_type)           \
  template <>                                                            \
  struct GetVectorPtrNameHelper<cpp_type> {                              \
    static const char* Call() { return "std_vector_" #cpp_type "_ptr"; } \
  };
PD_FOR_EACH_DATA_TYPE(SPECIALIZE_GetVectorPtrNameHelper);
#undef SPECIALIZE_GetVectorPtrNameHelper
template <>
struct GetVectorPtrNameHelper<std::string> {
  static const char* Call() { return "std_vector_str_ptr"; }
};

template <typename T>
struct TypeImpl<std::vector<T>*> : public std::monostate {
  using value_type = std::vector<T>*;

  const char* Name() const { return GetVectorPtrNameHelper<T>::Call(); }
};

}  // namespace ap::axpr
