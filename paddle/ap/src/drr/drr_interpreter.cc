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

#include "paddle/ap/include/drr/drr_interpreter.h"
#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/interpreter.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/drr/builtin_frame_util.h"
#include "paddle/ap/include/drr/drr_graph_descriptor.h"
#include "paddle/ap/include/drr/drr_node_descriptor.h"
#include "paddle/ap/include/drr/value.h"
#include "paddle/ap/include/drr/value_method_class.h"

namespace ap::drr {

namespace adt = ap::adt;

namespace {

using Function = ap::axpr::Function<ap::axpr::SerializableValue>;

using DrrNode = ap::drr::Node;
using DrrCtx = ap::drr::DrrCtx;

}  // namespace

DrrInterpreter::DrrInterpreter(
    const axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>&
        backend_ir_ctx)
    : interpreter_(ap::drr::MakeBuiltinFrameAttrMap(
          [&](const auto& Insert) { Insert(backend_ir_ctx); })) {}

adt::Result<DrrCtx> DrrInterpreter::InterpretDrrCtxMaker(
    const Function& lambda, const std::vector<axpr::Value>& args) {
  ADT_LET_CONST_REF(drr_ctx_val, interpreter_.Interpret(lambda, args));
  ADT_LET_CONST_REF(drr_ctx, drr_ctx_val.template CastTo<DrrCtx>())
      << adt::errors::TypeError{
             std::string() +
             "drr function should return a 'DrrCtx' object but '" +
             ap::axpr::GetTypeName(drr_ctx_val) + "' were given."};
  return drr_ctx;
}

adt::Result<DrrCtx> DrrInterpreter::InterpretPass(
    const Function& lambda, const std::string& drr_pass_name) {
  ADT_LET_CONST_REF(drr_ctx_val, interpreter_.Interpret(lambda, {}));
  ADT_LET_CONST_REF(drr_ctx, drr_ctx_val.template CastTo<DrrCtx>())
      << adt::errors::TypeError{
             std::string() +
             "drr function should return a 'DrrCtx' object but '" +
             ap::axpr::GetTypeName(drr_ctx_val) + "' were given."};
  return drr_ctx;
}

adt::Result<DrrCtx> DrrInterpreter::InterpretPass(
    const ap::axpr::ClassAttrs<ap::axpr::SerializableValue>& cls) {
  static ap::axpr::Lambda<ap::axpr::CoreExpr> lambda([] {
    ap::axpr::LambdaExprBuilder lmd;
    const ap::axpr::AnfExpr anf_expr = lmd.Lambda({"cls"}, [](auto& ctx) {
      auto& obj = ctx.Var("cls").Call();
      auto& method = obj.Attr("make_drr_ctx");
      auto& ret = method.Call();
      return ret;
    });
    const auto& core_expr = ap::axpr::ConvertAnfExprToCoreExpr(anf_expr);
    const auto& atomic = core_expr.Get<ap::axpr::Atomic<ap::axpr::CoreExpr>>();
    return atomic.Get<ap::axpr::Lambda<ap::axpr::CoreExpr>>();
  }());
  ap::axpr::Value cls_val{
      ap::axpr::TypeImpl<ap::axpr::ClassInstance<ap::axpr::Value>>(cls)};
  ADT_LET_CONST_REF(drr_ctx_val, interpreter_.Interpret(lambda, {cls_val}));
  ADT_LET_CONST_REF(drr_ctx, drr_ctx_val.template CastTo<DrrCtx>())
      << adt::errors::TypeError{
             std::string() +
             "drr function should return a 'DrrCtx' object but '" +
             ap::axpr::GetTypeName(drr_ctx_val) + "' were given."};
  return drr_ctx;
}

}  // namespace ap::drr
