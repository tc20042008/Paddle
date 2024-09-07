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
#include "paddle/phi/common/ap/arg_type.h"
#include "paddle/phi/common/ap/data_type.h"
#include "paddle/phi/common/ap/typed_buffer.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/pir/include/dialect/pexpr/value.h"

namespace phi {

class DenseTensor;

}

namespace ap::kernel_dispatch {

namespace adt = ::cinn::adt;

using kernel_define::ArgType;

using ArgValueImpl = std::variant<pexpr::DataValue, pexpr::PointerValue>;

struct ArgValue : public ArgValueImpl {
  using ArgValueImpl::ArgValueImpl;
  DEFINE_ADT_VARIANT_METHODS(ArgValueImpl);

  ArgType GetType() const {
    return Match([](auto impl) -> ArgType { return impl.GetType(); });
  }

  template <typename T>
  adt::Result<T> TryGet() const {
    if (!this->template Has<T>()) {
      return adt::errors::TypeError{
          std::string() + "ArgValue::TryGet() failed. T: " + typeid(T).name()};
    }
    return this->template Get<T>();
  }

  template <typename T>
  adt::Result<T> TryGetValue() const {
    if constexpr (std::is_pointer_v<T>) {
      const auto& pointer_value = this->template TryGet<pexpr::PointerValue>();
      ADT_RETURN_IF_ERROR(pointer_value);
      return pointer_value.GetOkValue().template TryGet<T>();
    } else {
      const auto& data_value = this->template TryGet<pexpr::DataValue>();
      ADT_RETURN_IF_ERROR(data_value);
      return data_value.GetOkValue().template TryGet<T>();
    }
  }
};

template <typename ValueT>
Result<ArgValue> CastToArgValue(const ValueT& value) {
  return value.Match(
      [&](const pexpr::DataValue& impl) -> Result<ArgValue> { return impl; },
      [&](const pexpr::PointerValue& impl) -> Result<ArgValue> { return impl; },
      [&](const auto&) -> Result<ArgValue> {
        return TypeError{std::string() +
                         "CastToArgValue failed. expected types: "
                         "(DataValue, PointerValue), actual type: " +
                         pexpr::MethodClass<ValueT>::Name(value)};
      });
}

}  // namespace ap::kernel_dispatch
