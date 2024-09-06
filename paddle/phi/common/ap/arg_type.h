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
#include "paddle/phi/common/ap/adt.h"
#include "paddle/phi/common/ap/data_type.h"
#include "paddle/pir/include/dialect/pexpr/value.h"
#include "paddle/pir/include/dialect/pexpr/value_method_class.h"

namespace phi {

class DenseTensor;

}

namespace ap::kernel_define {

using pexpr::ArithmeticType;
using pexpr::MethodClass;
using pexpr::PointerType;

using ArgTypeImpl = std::variant<ArithmeticType, PointerType>;

struct ArgType : public ArgTypeImpl {
  using ArgTypeImpl::ArgTypeImpl;
  DEFINE_ADT_VARIANT_METHODS(ArgTypeImpl);

  const char* Name() const {
    return Match([](const auto& impl) { return impl.Name(); });
  }

  template <typename T>
  adt::Result<T> TryGet() const {
    if (!this->template Has<T>()) {
      return adt::errors::TypeError{
          std::string() + "ArgType::TryGet() failed. T: " + typeid(T).name()};
    }
    return this->template Get<T>();
  }

  template <typename T>
  bool IsType() const {
    if constexpr (std::is_pointer_v<T>) {
      const auto& pointer_type = this->template TryGet<pexpr::PointerType>();
      if (pointer_type.HasError()) {
        return false;
      }
      return pointer_type.GetOkValue().template Has<pexpr::CppPointerType<T>>();
    } else {
      const auto& arithmetic_type =
          this->template TryGet<pexpr::ArithmeticType>();
      if (arithmetic_type.HasError()) {
        return false;
      }
      return arithmetic_type.GetOkValue()
          .template Has<pexpr::CppArithmeticType<T>>();
    }
  }
};

template <typename ValueT>
Result<ArgType> CastToArgType(const ValueT& val) {
  return val.Match(
      [&](const ArithmeticType& atype) -> Result<ArgType> {
        return ArgType{atype};
      },
      [&](const PointerType& ptype) -> Result<ArgType> {
        return ArgType{ptype};
      },
      [&](const auto&) -> Result<ArgType> {
        return adt::errors::TypeError{
            std::string() +
            "CastToArgType failed. expected types: "
            "(ArithmeticType, PointerType), actual type: " +
            MethodClass<ValueT>::Name(val)};
      });
}

}  // namespace ap::kernel_define
