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
#include "paddle/pir/include/dialect/pexpr/pointer_type.h"

namespace pexpr {

using PointerValueImpl = std::variant<
#define MAKE_ARG_VALUE_ALTERNATIVE(cpp_type, enum_type) \
  cpp_type*, const cpp_type*,
    PD_FOR_EACH_DATA_TYPE(MAKE_ARG_VALUE_ALTERNATIVE) void*,
    const void*
#undef MAKE_ARG_VALUE_ALTERNATIVE
    >;

struct PointerValue : public PointerValueImpl {
  using PointerValueImpl::PointerValueImpl;
  DEFINE_ADT_VARIANT_METHODS(PointerValueImpl);

  PointerType GetType() const {
    return Match([](auto impl) -> PointerType {
      return PointerType{CppPointerType<decltype(impl)>{}};
    });
  }

  template <typename T>
  Result<T> TryGet() const {
    if (!this->template Has<T>()) {
      return TypeError{
          std::string() + "PointerValue::TryGet() failed. expected_type: " +
          CppPointerType<T>{}.Name() + ", actual_type: " + GetType().Name()};
    }
    return this->template Get<T>();
  }
};

}  // namespace pexpr
