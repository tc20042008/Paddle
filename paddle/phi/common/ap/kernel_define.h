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
#include <cstdint>
#include "paddle/phi/common/ap/adt.h"

namespace ap {

template <typename T>
struct ArgType {};

#define __BOOL bool
#define __CHAR char
#define __FLOAT float
#define __DOUBLE double

// clang-format off
#define AP_FOR_EACH_CPP_TYPE(__macro)   \
  __macro(__BOOL)                       \
  __macro(__CHAR)                       \
  __macro(uint8_t)                      \
  __macro(int16_t)                      \
  __macro(uint16_t)                     \
  __macro(int32_t)                      \
  __macro(int64_t)                      \
  __macro(uint32_t)                     \
  __macro(uint64_t)                     \
  __macro(__FLOAT)                      \
  __macro(__DOUBLE)
// clang-format on

// clang-format off
using KernelArgTypeImpl = std::variant<
#define MAKE_CPP_TYPE_ALTERNATIVE(cpp_type)   \
    ArgType<cpp_type>,                        \
    ArgType<const cpp_type>,                  \
    ArgType<cpp_type*>,                       \
    ArgType<const cpp_type*>,
    AP_FOR_EACH_CPP_TYPE(MAKE_CPP_TYPE_ALTERNATIVE)
#undef MAKE_CPP_TYPE_ALTERNATIVE
    ArgType<void*>,
    ArgType<const void*>>;
// clang-format on

struct KernelArgType : public KernelArgTypeImpl {
  using KernelArgTypeImpl::KernelArgTypeImpl;
  DEFINE_ADT_VARIANT_METHODS(KernelArgTypeImpl);
};

}  // namespace ap
