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

#include "paddle/phi/core/ap/kernel_define_helper.h"
#include "ap/axpr/cps_expr_interpreter.h"
#include "ap/kernel_define/runtime_value.h"
#include "ap/kernel_define/runtime_value_method_class.h"

namespace phi {

namespace {

using CoreExpr = ap::axpr::CoreExpr;

using Lambda = ap::axpr::Lambda<CoreExpr>;

using Module = ap::kernel_define::Module;

using IrNodeT = ap::kernel_define::UndefinedIrNode;

using Val = ap::kernel_define::RtValue<IrNodeT>;

}  // namespace

adt::Result<Module> KernelDefineHelper::InterpretKernelDefineLambda(
    const Lambda& lambda) {
  ap::axpr::CpsExprInterpreter<Val> cps_interpreter{};
  adt::Nothing none{};
  ADT_LET_CONST_REF(interpret_ret, cps_interpreter.Interpret(lambda, {none}));
  ADT_LET_CONST_REF(m, interpret_ret.TryGet<Module>());
  return m;
}

}  // namespace phi
