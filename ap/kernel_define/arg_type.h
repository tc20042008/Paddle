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
#include "ap/axpr/value.h"
#include "ap/axpr/value_method_class.h"
#include "ap/kernel_define/adt.h"
#include "ap/kernel_define/data_type.h"

namespace phi {

class DenseTensor;

}

namespace ap::kernel_define {

using ap::axpr::DataType;
using ap::axpr::MethodClass;
using ap::axpr::PointerType;

using ArgTypeImpl = std::variant<DataType, PointerType>;

struct ArgType : public ArgTypeImpl {
  using ArgTypeImpl::ArgTypeImpl;
  DEFINE_ADT_VARIANT_METHODS(ArgTypeImpl);

  const char* Name() const {
    return Match([](const auto& impl) { return impl.Name(); });
  }

  template <typename T>
  bool IsType() const {
    if constexpr (std::is_pointer_v<T>) {
      const auto& pointer_type = this->template TryGet<ap::axpr::PointerType>();
      if (pointer_type.HasError()) {
        return false;
      }
      return pointer_type.GetOkValue()
          .template Has<ap::axpr::CppPointerType<T>>();
    } else {
      const auto& data_type = this->template TryGet<ap::axpr::DataType>();
      if (data_type.HasError()) {
        return false;
      }
      return data_type.GetOkValue().template Has<ap::axpr::CppDataType<T>>();
    }
  }
};

template <typename ValueT>
Result<ArgType> CastToArgType(const ValueT& val) {
  return val.Match(
      [&](const DataType& atype) -> Result<ArgType> { return ArgType{atype}; },
      [&](const PointerType& ptype) -> Result<ArgType> {
        return ArgType{ptype};
      },
      [&](const auto&) -> Result<ArgType> {
        return adt::errors::TypeError{std::string() +
                                      "CastToArgType failed. expected types: "
                                      "(DataType, PointerType), actual type: " +
                                      MethodClass<ValueT>::Name(val)};
      });
}

}  // namespace ap::kernel_define
