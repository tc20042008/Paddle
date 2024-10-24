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

#include "ap/axpr/data_type_util.h"
#include "ap/axpr/method_class.h"
#include "ap/paddle/const_meta_tensor_ptr.h"

namespace ap::paddle {

template <typename ValueT>
struct ConstMetaTensorPtrMethodClass {
  using This = ConstMetaTensorPtrMethodClass;
  using Self = ConstMetaTensorPtr;

  adt::Result<ValueT> ToString(const Self& self) {
    std::ostringstream ss;
    const auto* ptr = self;
    ss << "<" << axpr::TypeImpl<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    if (attr_name == "dtype") {
      return GetDtype(self);
    }
    if (attr_name == "dims") {
      return GetDims(self);
    }
    return adt::errors::AttributeError{
        std::string() + "'ConstMetaTensorPtr' object has no attribute '" +
        attr_name + "'."};
  }

  adt::Result<ValueT> GetDims(const Self& self) { return self->dims(); }

  adt::Result<ValueT> GetDtype(const Self& self) {
    ADT_LET_CONST_REF(dtype, axpr::GetDataTypeFromPhiDataType(self->dtype()));
    return dtype;
  }
};

}  // namespace ap::paddle

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, paddle::ConstMetaTensorPtr>
    : public paddle::ConstMetaTensorPtrMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<paddle::ConstMetaTensorPtr>> {};

}  // namespace ap::axpr
