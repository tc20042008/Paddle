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

#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/ap_drr_helper.h"
#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/interpreter.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/drr/builtin_frame_util.h"
#include "paddle/ap/include/drr/drr_graph_descriptor.h"
#include "paddle/ap/include/drr/drr_node_descriptor.h"
#include "paddle/ap/include/drr/value.h"
#include "paddle/ap/include/drr/value_method_class.h"

namespace cinn::dialect::ir {

namespace {

using Function = ap::axpr::Function<ap::axpr::SerializableValue>;

using DrrNode = ap::drr::Node;
using DrrCtx = ap::drr::DrrCtx;

}  // namespace

adt::Result<DrrCtx> ApDrrHelper::Interpret(const Function& lambda,
                                           const std::string& drr_pass_name) {
  ap::axpr::Interpreter interpreter(ap::drr::MakeBuiltinFrameAttrMap());
  ADT_LET_CONST_REF(drr_ctx_val, interpreter.Interpret(lambda, {}));
  ADT_LET_CONST_REF(drr_ctx, drr_ctx_val.template CastTo<DrrCtx>())
      << adt::errors::TypeError{
             std::string() +
             "drr function should return a 'DrrCtx' object but '" +
             ap::axpr::GetTypeName(drr_ctx_val) + "' were given."};
  return drr_ctx;
}

adt::Result<DrrCtx> ApDrrHelper::Interpret(
    const ap::registry::DrrPassRegistryItem& item) {
  static ap::axpr::Lambda<ap::axpr::CoreExpr> lambda([] {
    ap::axpr::LambdaExprBuilder lmd;
    const ap::axpr::AnfExpr anf_expr = lmd.Lambda({"cls"}, [](auto& ctx) {
      return ctx.Var("cls").Call().Attr("make_drr_ctx").Call();
    });
    const auto& core_expr = ap::axpr::ConvertAnfExprToCoreExpr(anf_expr);
    const auto& atomic = core_expr.Get<ap::axpr::Atomic<ap::axpr::CoreExpr>>();
    return atomic.Get<ap::axpr::Lambda<ap::axpr::CoreExpr>>();
  }());
  ap::axpr::Interpreter interpreter(ap::drr::MakeBuiltinFrameAttrMap());
  ap::axpr::Value cls{
      ap::axpr::TypeImpl<ap::axpr::ClassInstance<ap::axpr::Value>>(item->cls)};
  ADT_LET_CONST_REF(drr_ctx_val, interpreter.Interpret(lambda, {cls}));
  ADT_LET_CONST_REF(drr_ctx, drr_ctx_val.template CastTo<DrrCtx>())
      << adt::errors::TypeError{
             std::string() +
             "drr function should return a 'DrrCtx' object but '" +
             ap::axpr::GetTypeName(drr_ctx_val) + "' were given."};
  return drr_ctx;
}

}  // namespace cinn::dialect::ir
