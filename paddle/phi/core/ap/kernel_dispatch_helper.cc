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

#include "paddle/phi/core/ap/kernel_dispatch_helper.h"
#include "ap/axpr/cps_expr_interpreter.h"
#include "ap/kernel_dispatch/dispatch_ctx_value.h"
#include "ap/kernel_dispatch/dispatch_ctx_value_method_class.h"

namespace phi {

namespace {

using CoreExpr = ap::axpr::CoreExpr;
using Lambda = ap::axpr::Lambda<CoreExpr>;
using Val = ap::kernel_dispatch::Val;
using DispatchRawCtx = ap::kernel_dispatch::DispatchRawCtx<Val>;

}  // namespace

adt::Result<adt::Ok> KernelDispatchHelper::Interpret(
    const Lambda& kernel_dispatcher_lambda,
    const Lambda& ctx_maker_lambda,
    const DispatchRawCtx& raw_ctx) {
  ap::axpr::CpsExprInterpreter<Val> cps_interpreter{};
  ADT_LET_CONST_REF(ctx,
                    cps_interpreter.Interpret(ctx_maker_lambda, {raw_ctx}));
  ADT_RETURN_IF_ERR(cps_interpreter.Interpret(kernel_dispatcher_lambda, {ctx}));
  return adt::Ok{};
}

}  // namespace phi
