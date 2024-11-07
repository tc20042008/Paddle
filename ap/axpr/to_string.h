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
#include "ap/axpr/method_class.h"

namespace ap::axpr {

template <typename ValueT>
adt::Result<std::string> ToString(const ValueT& val) {
  const auto& unary_func = MethodClass<ValueT>::ToString(val);
  ADT_LET_CONST_REF(str_val, unary_func(val));
  ADT_LET_CONST_REF(str, str_val.template TryGet<std::string>());
  return str;
}

template <typename ValueT>
std::string ToDebugString(const ValueT& val) {
  const auto& str = ToString(val);
  if (str.HasError()) {
    return "[invalid debug string]";
  }
  return str.GetOkValue();
}

}  // namespace ap::axpr
