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
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/pstring.h"

namespace pexpr {

using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;
using float16 = ::phi::dtype::float16;
using bfloat16 = ::phi::dtype::bfloat16;
using float8_e4m3fn = ::phi::dtype::float8_e4m3fn;
using float8_e5m2 = ::phi::dtype::float8_e5m2;
using pstring = ::phi::dtype::pstring;

#define PEXPR_FOR_EACH_INT_TYPE(_) \
  _(int8)                          \
  _(uint8)                         \
  _(int16)                         \
  _(uint16)                        \
  _(int32)                         \
  _(uint32)                        \
  _(int64)                         \
  _(uint64)

#define PEXPR_FOR_EACH_ARITHMETIC_OP_SUPPORTED_TYPE(_) \
  _(bool)                                              \
  _(float)                                             \
  _(double)                                            \
  _(int8_t)                                            \
  _(uint8_t)                                           \
  _(int16_t)                                           \
  _(uint16_t)                                          \
  _(int32_t)                                           \
  _(uint32_t)                                          \
  _(int64_t)                                           \
  _(uint64_t)

namespace detail {

template <typename T>
struct IsArithmeticOpSupportedHelper {
  static constexpr bool value = false;
};

#define SPECIALIZE_IS_ARITHMETIC_OP_SUPPORTED(cpp_type) \
  template <>                                           \
  struct IsArithmeticOpSupportedHelper<cpp_type> {      \
    static constexpr bool value = true;                 \
  };

PEXPR_FOR_EACH_ARITHMETIC_OP_SUPPORTED_TYPE(
    SPECIALIZE_IS_ARITHMETIC_OP_SUPPORTED)

#undef SPECIALIZE_IS_ARITHMETIC_OP_SUPPORTED

}  // namespace detail

template <typename T>
constexpr bool IsArithmeticOpSupported() {
  return detail::IsArithmeticOpSupportedHelper<T>::value;
}

}  // namespace pexpr
