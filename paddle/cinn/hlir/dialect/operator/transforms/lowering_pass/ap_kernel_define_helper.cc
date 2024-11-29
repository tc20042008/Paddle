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
#include "ap/axpr/cps_interpreter.h"
#include "ap/code_gen/builtin_frame_util.h"
#include "ap/code_gen/value.h"
#include "ap/code_gen/value_method_class.h"
#include "ap/drr/drr_graph_descriptor.h"
#include "ap/drr/drr_node_descriptor.h"
#include "ap/paddle/op_cuda_code_gen_impl.h"
#include "ap/paddle/pir_node_method_class.h"

namespace cinn::dialect::ir {

namespace {

using Function = ap::axpr::Function<ap::axpr::SerializableValue>;
using Module = ap::code_module::Module;
using PirNode = ap::paddle::PirNode;
using Val = ap::code_gen::Value<PirNode>;
using CodeGenCtx = ap::code_gen::CodeGenCtx<PirNode>;
using CodeGenResult = ap::code_gen::CodeGenResult<Val>;

}  // namespace

adt::Result<CodeGenResult> ApKernelDefineHelper::Interpret(
    const Function& lambda, const CodeGenCtx& code_gen_ctx) {
  ap::axpr::CpsInterpreter<Val> interpreter(
      ap::code_gen::MakeBuiltinFrameAttrMap<Val>());
  ADT_CHECK(code_gen_ctx->ir_match_ctx.has_value());
  const auto& ir_match_ctx = code_gen_ctx->ir_match_ctx.value();
  ap::ir_match::OpMatchCtx<PirNode> op_match_ctx{ir_match_ctx.shared_ptr()};
  ap::ir_match::TensorMatchCtx<PirNode> tensor_match_ctx{
      ir_match_ctx.shared_ptr()};
  ADT_LET_CONST_REF(
      result,
      interpreter.Interpret(lambda,
                            {code_gen_ctx, op_match_ctx, tensor_match_ctx}));
  ADT_LET_CONST_REF(m, result.template TryGet<CodeGenResult>());
  return m;
}

}  // namespace cinn::dialect::ir
