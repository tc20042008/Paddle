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

#include "ap/axpr/constants.h"
#include "ap/axpr/method_class.h"
#include "ap/axpr/object.h"

namespace ap::axpr {

template <typename ValueT>
struct ObjectMethodClass {
  using This = ObjectMethodClass;
  using Self = Object<ValueT>;

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    ADT_LET_CONST_REF(val, self->Get(attr_name)) << adt::errors::AttributeError{
        std::string() + "'object' has no attribute '" + attr_name + "'."};
    return val;
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, Object<ValueT>>
    : public ObjectMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<Object<ValueT>>>
    : public EmptyMethodClass<ValueT> {};

}  // namespace ap::axpr
