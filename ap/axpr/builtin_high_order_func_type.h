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

#include "ap/axpr/adt.h"
#include "ap/axpr/core_expr.h"
#include "ap/axpr/error.h"
#include "ap/axpr/type.h"

namespace ap::axpr {

template <typename ValueT>
using ApplyT = std::function<Result<ValueT>(const ValueT& func,
                                            const std::vector<ValueT>& args)>;

template <typename ValueT>
using BuiltinHighOrderFuncType =
    Result<ValueT> (*)(const ApplyT<ValueT>& apply,
                       const ValueT& obj,
                       const std::vector<ValueT>& args);

template <typename ValueT>
struct TypeImpl<BuiltinHighOrderFuncType<ValueT>> : public std::monostate {
  using value_type = BuiltinHighOrderFuncType<ValueT>;

  const char* Name() const { return "builtin_high_order_function"; }
};

}  // namespace ap::axpr
