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
#include "ap/axpr/bool.h"
#include "ap/axpr/builtin_func_type.h"
#include "ap/axpr/builtin_high_order_func_type.h"
#include "ap/axpr/builtin_symbol.h"
#include "ap/axpr/closure.h"
#include "ap/axpr/cps_builtin_high_order_func_type.h"
#include "ap/axpr/data_type.h"
#include "ap/axpr/data_value.h"
#include "ap/axpr/environment.h"
#include "ap/axpr/environment_mgr.h"
#include "ap/axpr/error.h"
#include "ap/axpr/float.h"
#include "ap/axpr/frame.h"
#include "ap/axpr/int.h"
#include "ap/axpr/lambda.h"
#include "ap/axpr/list.h"
#include "ap/axpr/method.h"
#include "ap/axpr/nothing.h"
#include "ap/axpr/object.h"
#include "ap/axpr/packed_args.h"
#include "ap/axpr/pointer_type.h"
#include "ap/axpr/pointer_value.h"
#include "ap/axpr/starred.h"
#include "ap/axpr/string.h"
#include "ap/axpr/type.h"
#include "ap/axpr/type_util.h"

namespace ap::axpr {

using adt::Nothing;

template <typename ValueT, typename... Ts>
using ValueBase = std::variant<Type<Nothing,
                                    bool,
                                    int64_t,
                                    double,
                                    std::string,
                                    adt::List<ValueT>,
                                    axpr::Object<ValueT>,
                                    PackedArgs<ValueT>,
                                    Lambda<CoreExpr>,
                                    Closure<ValueT>,
                                    Method<ValueT>,
                                    builtin_symbol::Symbol,
                                    Starred<ValueT>,
                                    BuiltinFuncType<ValueT>,
                                    BuiltinHighOrderFuncType<ValueT>,
                                    CpsBuiltinHighOrderFuncType<ValueT>,
                                    Ts...>,
                               Nothing,
                               bool,
                               int64_t,
                               double,
                               std::string,
                               adt::List<ValueT>,
                               axpr::Object<ValueT>,
                               PackedArgs<ValueT>,
                               Lambda<CoreExpr>,
                               Closure<ValueT>,
                               Method<ValueT>,
                               builtin_symbol::Symbol,
                               Starred<ValueT>,
                               BuiltinFuncType<ValueT>,
                               BuiltinHighOrderFuncType<ValueT>,
                               CpsBuiltinHighOrderFuncType<ValueT>,
                               Ts...>;

template <typename ValueT>
using Builtin = ValueBase<ValueT>;

template <typename ValueT>
ValueT GetType(const ValueT& value) {
  return value.Match([](const auto& impl) -> ValueT {
    return TypeImpl<std::decay_t<decltype(impl)>>{};
  });
}

}  // namespace ap::axpr
