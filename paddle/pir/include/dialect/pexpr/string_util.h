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

#include <string>
#include "paddle/pir/include/dialect/pexpr/arithmetic_value.h"
#include "paddle/pir/include/dialect/pexpr/constants.h"
#include "paddle/pir/include/dialect/pexpr/method_class.h"

namespace pexpr {

template <typename ArithmeticOp, typename Val, typename T>
struct BuiltinStringBinaryHelper {
  static Result<Val> Call(const std::string& str, const T& rhs) {
    return adt::errors::TypeError{std::string() +
                                  "unsupported operand types for " +
                                  ArithmeticOp::Name() + ": 'str' and '" +
                                  MethodClassImpl<Val, T>::Name() + "'"};
  }
};

template <typename Val>
struct BuiltinStringBinaryHelper<ArithmeticAdd, Val, std::string> {
  static Result<Val> Call(const std::string& lhs, const std::string& rhs) {
    return ArithmeticAdd::Call(lhs, rhs);
  }
};

template <typename Val>
struct BuiltinStringBinaryHelper<ArithmeticAdd, Val, ArithmeticValue> {
  static Result<Val> Call(const std::string& lhs,
                          const ArithmeticValue& rhs_val) {
    const auto& rhs = rhs_val.Match([](auto impl) -> Result<std::string> {
      using T = decltype(impl);
      if constexpr (IsArithmeticOpSupported<T>()) {
        return std::to_string(impl);
      } else {
        return adt::errors::TypeError{std::string() +
                                      "unsupported operand types for " +
                                      ArithmeticAdd::Name() + ": 'str' and '" +
                                      CppArithmeticType<T>{}.Name() + "'"};
      }
    });
    ADT_RETURN_IF_ERROR(rhs);
    return lhs + rhs.GetOkValue();
  }
};

#define SPECIALIZE_BuiltinStringBinaryHelper_string_cmp(cls_name)             \
  template <typename Val>                                                     \
  struct BuiltinStringBinaryHelper<cls_name, Val, std::string> {              \
    static Result<Val> Call(const std::string& lhs, const std::string& rhs) { \
      return ArithmeticValue{cls_name::Call(lhs, rhs)};                       \
    }                                                                         \
  };
SPECIALIZE_BuiltinStringBinaryHelper_string_cmp(ArithmeticEQ);
SPECIALIZE_BuiltinStringBinaryHelper_string_cmp(ArithmeticNE);
SPECIALIZE_BuiltinStringBinaryHelper_string_cmp(ArithmeticGT);
SPECIALIZE_BuiltinStringBinaryHelper_string_cmp(ArithmeticGE);
SPECIALIZE_BuiltinStringBinaryHelper_string_cmp(ArithmeticLT);
SPECIALIZE_BuiltinStringBinaryHelper_string_cmp(ArithmeticLE);
#undef SPECIALIZE_BuiltinStringBinaryHelper_string

template <typename Val>
struct BuiltinStringBinaryHelper<ArithmeticMul, Val, ArithmeticValue> {
  static Result<Val> Call(const std::string& lhs,
                          const ArithmeticValue& rhs_val) {
    const auto& opt_uint64 =
        rhs_val.StaticCastTo(CppArithmeticType<uint64_t>{});
    ADT_RETURN_IF_ERROR(opt_uint64);
    const auto& opt_size = opt_uint64.GetOkValue().template TryGet<uint64_t>();
    ADT_RETURN_IF_ERROR(opt_size);
    uint64_t size = opt_size.GetOkValue();
    std::ostringstream ss;
    for (int i = 0; i < size; ++i) {
      ss << lhs;
    }
    return ss.str();
  }
};

template <typename ArithmeticOp, typename Val>
Result<Val> BuiltinStringBinary(const std::string& str, const Val& rhs_val) {
  return rhs_val.Match([&](const auto& rhs) -> Result<Val> {
    using T = std::decay_t<decltype(rhs)>;
    return BuiltinStringBinaryHelper<ArithmeticOp, Val, T>::Call(str, rhs);
  });
}

}  // namespace pexpr
