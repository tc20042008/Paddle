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
#include "ap/paddle/ddim.h"
#include "ap/paddle/meta_tensor_ptr.h"

namespace ap::paddle {

template <typename ValueT>
struct MetaTensorPtrMethodClass {
  using This = MetaTensorPtrMethodClass;
  using Self = MetaTensorPtr;

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
        std::string() + "'MetaTensorPtr' object has no attribute '" +
        attr_name + "'."};
  }

  adt::Result<ValueT> GetDims(const Self& self) { return self->dims(); }

  adt::Result<ValueT> GetDtype(const Self& self) {
    ADT_LET_CONST_REF(dtype, axpr::GetDataTypeFromPhiDataType(self->dtype()));
    return dtype;
  }

  adt::Result<ValueT> SetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    if (attr_name == "dtype") {
      return axpr::Method<ValueT>{self, &This::StaticSetDtype};
    }
    if (attr_name == "dims") {
      return axpr::Method<ValueT>{self, &This::StaticSetDims};
    }
    return adt::errors::AttributeError{
        std::string() + "'MetaTensorPtr' object has no attribute '" +
        attr_name + "'."};
  }

  static adt::Result<ValueT> StaticSetDtype(const ValueT& self_val,
                                            const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template TryGet<Self>());
    ADT_CHECK(args.size() == 2);
    ADT_LET_CONST_REF(data_type, args.at(1).template TryGet<axpr::DataType>());
    ADT_LET_CONST_REF(dtype, GetPhiDataTypeFromDataType(data_type));
    self->set_dtype(dtype);
    return adt::Nothing{};
  }

  static adt::Result<ValueT> StaticSetDims(const ValueT& self_val,
                                           const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template TryGet<Self>());
    ADT_CHECK(args.size() == 2);
    return This{}.SetDims(self, args.at(1));
  }

  adt::Result<ValueT> SetDims(const Self& self, const ValueT& dims_val) {
    return dims_val.Match(
        [&](const DDim& ddims) -> adt::Result<ValueT> {
          return SetDimsByDDim(self, ddims);
        },
        [&](const adt::List<ValueT>& list) -> adt::Result<ValueT> {
          return SetDimsByIntList(self, list);
        },
        [&](const auto&) -> adt::Result<ValueT> {
          return adt::errors::TypeError{"only DDim or list of int supported."};
        });
  }

  adt::Result<ValueT> SetDimsByDDim(const Self& self, const DDim& ddims) {
    self->set_dims(ddims);
    return adt::Nothing{};
  }

  adt::Result<ValueT> SetDimsByIntList(const Self& self,
                                       const adt::List<ValueT>& list) {
    std::vector<int64_t> dims{};
    dims.reserve(list->size());
    for (const auto& dim_val : *list) {
      ADT_LET_CONST_REF(dim, dim_val.template TryGet<int64_t>());
      dims.push_back(dim);
    }
    self->set_dims(::common::make_ddim(dims));
    return adt::Nothing{};
  }
};

}  // namespace ap::paddle

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, paddle::MetaTensorPtr>
    : public paddle::MetaTensorPtrMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<paddle::MetaTensorPtr>> {};

}  // namespace ap::axpr
