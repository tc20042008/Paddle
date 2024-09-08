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
#include "paddle/pir/include/dialect/pexpr/atomic.h"
#include "paddle/pir/include/dialect/pexpr/core_expr.h"
#include "paddle/pir/include/dialect/pexpr/error.h"
#include "paddle/pir/include/dialect/pexpr/type.h"

namespace pexpr {

template <typename ValueT>
class Environment;

template <typename ValueT>
class CpsInterpreterBase {
 public:
  virtual Result<ValueT> Interpret(const Closure<ValueT>& closure,
                                   const std::vector<ValueT>& args) = 0;

  virtual Result<adt::Ok> InterpretLambdaCall(
      const std::shared_ptr<Environment<ValueT>>& env,
      const ValueT& outter_func,
      const Lambda<CoreExpr>& lambda,
      const std::vector<ValueT>& args,
      ComposedCallImpl<ValueT>* ret_composed_call) = 0;
};

template <typename ValueT>
using CpsBuiltinHighOrderFuncType =
    Result<adt::Ok> (*)(CpsInterpreterBase<ValueT>* CpsInterpret,
                        ComposedCallImpl<ValueT>* composed_call);

template <typename ValueT>
struct TypeImpl<CpsBuiltinHighOrderFuncType<ValueT>> : public std::monostate {
  using value_type = CpsBuiltinHighOrderFuncType<ValueT>;

  const char* Name() const { return "cps_builtin_function"; }
};

}  // namespace pexpr
