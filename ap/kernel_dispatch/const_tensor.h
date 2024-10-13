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
#include "ap/adt/adt.h"
#include "ap/axpr/type.h"
#include "ap/kernel_define/data_type.h"
#include "ap/kernel_dispatch/typed_buffer.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

class DenseTensor;

}

namespace ap::kernel_dispatch {

using ConstTensorDataImpl = std::variant<const phi::DenseTensor*, TypedBuffer>;
struct ConstTensorData : public ConstTensorDataImpl {
  using ConstTensorDataImpl::ConstTensorDataImpl;
  DEFINE_ADT_VARIANT_METHODS(ConstTensorDataImpl);

  template <typename T>
  const T* data() const {
    return Match(
        [](const phi::DenseTensor* tensor) -> const T* {
          return reinterpret_cast<const T*>(tensor->data());
        },
        [](const TypedBuffer& buffer) -> const T* {
          return reinterpret_cast<T*>(buffer->buffer);
        });
  }
  const void* data() const {
    return Match(
        [](const phi::DenseTensor* tensor) -> const void* {
          return tensor->data();
        },
        [](const TypedBuffer& buffer) -> const void* {
          return buffer->buffer;
        });
  }
  phi::DataType dtype() const {
    return Match([](const phi::DenseTensor* tensor) { return tensor->dtype(); },
                 [](const TypedBuffer& buffer) { return buffer->dtype; });
  }
};

template <typename ValueT>
struct ConstTensorImpl {
  ConstTensorData tensor_data;
  adt::List<ValueT> dims;

  bool operator==(const ConstTensorImpl& other) const {
    return other.tensor_data == this->tensor_data && other.dims == this->dims;
  }
};

template <typename ValueT>
DEFINE_ADT_RC(ConstTensor, ConstTensorImpl<ValueT>);

}  // namespace ap::kernel_dispatch

namespace ap::axpr {

template <typename ValueT>
struct TypeImpl<ap::kernel_dispatch::ConstTensor<ValueT>>
    : public std::monostate {
  using value_type = ap::kernel_dispatch::ConstTensor<ValueT>;

  const char* Name() const { return "ConstTensor"; }
};

}  // namespace ap::axpr
