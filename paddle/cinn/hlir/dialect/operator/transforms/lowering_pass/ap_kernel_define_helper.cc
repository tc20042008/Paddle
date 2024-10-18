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

#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/ap_kernel_define_helper.h"
#include "ap/axpr/cps_expr_interpreter.h"
#include "ap/drr/drr_graph_descriptor.h"
#include "ap/drr/drr_node_descriptor.h"
#include "ap/kernel_define/compiletime_value.h"
#include "ap/kernel_define/compiletime_value_method_class.h"
#include "ap/paddle/op_cuda_code_gen_impl.h"
#include "ap/paddle/pir_node_method_class.h"

namespace cinn::dialect::ir {

namespace {

using CoreExpr = ap::axpr::CoreExpr;
using Lambda = ap::axpr::Lambda<CoreExpr>;
using Module = ap::kernel_define::Module;
using PirNode = ap::paddle::PirNode;
using DefineCtx = ap::kernel_define::DefineCtx<PirNode>;
using Val = ap::kernel_define::CtValue<PirNode>;

}  // namespace

adt::Result<Module> ApKernelDefineHelper::Interpret(
    const Lambda& lambda, const DefineCtx& define_ctx) {
  ap::axpr::CpsExprInterpreter<Val> interpreter;
  ADT_CHECK(define_ctx->ir_match_ctx.has_value());
  const auto& ir_match_ctx = define_ctx->ir_match_ctx.value();
  ap::ir_match::OpMatchCtx<PirNode> op_match_ctx{ir_match_ctx.shared_ptr()};
  ap::ir_match::TensorMatchCtx<PirNode> tensor_match_ctx{
      ir_match_ctx.shared_ptr()};
  ADT_LET_CONST_REF(module_val,
                    interpreter.Interpret(
                        lambda, {define_ctx, op_match_ctx, tensor_match_ctx}));
  ADT_LET_CONST_REF(m, module_val.template TryGet<Module>());
  return m;
}

}  // namespace cinn::dialect::ir
