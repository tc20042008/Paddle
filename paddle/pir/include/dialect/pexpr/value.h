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

#include "paddle/pir/include/dialect/pexpr/adt.h"
#include "paddle/pir/include/dialect/pexpr/bool.h"
#include "paddle/pir/include/dialect/pexpr/builtin_func_type.h"
#include "paddle/pir/include/dialect/pexpr/builtin_symbol.h"
#include "paddle/pir/include/dialect/pexpr/closure.h"
#include "paddle/pir/include/dialect/pexpr/cps_builtin_high_order_func_type.h"
#include "paddle/pir/include/dialect/pexpr/data_type.h"
#include "paddle/pir/include/dialect/pexpr/data_value.h"
#include "paddle/pir/include/dialect/pexpr/environment.h"
#include "paddle/pir/include/dialect/pexpr/environment_mgr.h"
#include "paddle/pir/include/dialect/pexpr/error.h"
#include "paddle/pir/include/dialect/pexpr/float.h"
#include "paddle/pir/include/dialect/pexpr/frame.h"
#include "paddle/pir/include/dialect/pexpr/int.h"
#include "paddle/pir/include/dialect/pexpr/list.h"
#include "paddle/pir/include/dialect/pexpr/method.h"
#include "paddle/pir/include/dialect/pexpr/nothing.h"
#include "paddle/pir/include/dialect/pexpr/object.h"
#include "paddle/pir/include/dialect/pexpr/pointer_type.h"
#include "paddle/pir/include/dialect/pexpr/pointer_value.h"
#include "paddle/pir/include/dialect/pexpr/string.h"
#include "paddle/pir/include/dialect/pexpr/type.h"

namespace pexpr {

using adt::Nothing;

template <typename ValueT, typename... Ts>
using ValueBase = std::variant<Type<Nothing,
                                    bool,
                                    int64_t,
                                    double,
                                    std::string,
                                    Closure<ValueT>,
                                    Method<ValueT>,
                                    adt::List<ValueT>,
                                    builtin_symbol::Symbol,
                                    BuiltinFuncType<ValueT>,
                                    CpsBuiltinHighOrderFuncType<ValueT>,
                                    Ts...>,
                               Nothing,
                               bool,
                               int64_t,
                               double,
                               std::string,
                               Closure<ValueT>,
                               Method<ValueT>,
                               adt::List<ValueT>,
                               builtin_symbol::Symbol,
                               BuiltinFuncType<ValueT>,
                               CpsBuiltinHighOrderFuncType<ValueT>,
                               Ts...>;

template <typename ValueT>
using Builtin = ValueBase<ValueT>;

}  // namespace pexpr
