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

#include "ap/axpr/builtin_func_type.h"
#include "ap/axpr/constants.h"
#include "ap/axpr/method_class.h"

namespace ap::axpr {

template <typename ValueT>
struct BuiltinFuncTypeMethodClass {
  using This = BuiltinFuncTypeMethodClass;
  using Self = BuiltinFuncType<ValueT>;

  adt::Result<ValueT> ToString(Self func) {
    std::ostringstream ss;
    ss << "<" << TypeImpl<Self>{}.Name() << " object at " << func << ">";
    return ss.str();
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, BuiltinFuncType<ValueT>>
    : public BuiltinFuncTypeMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<BuiltinFuncType<ValueT>>>
    : public EmptyMethodClass<ValueT> {};

}  // namespace ap::axpr
