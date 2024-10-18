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
#include "ap/axpr/cps_expr_interpreter.h"
#include "ap/drr/drr_graph_descriptor.h"
#include "ap/drr/drr_node_descriptor.h"
#include "ap/drr/drr_value.h"
#include "ap/drr/drr_value_method_class.h"

namespace cinn::dialect::ir {

namespace {

using CoreExpr = ap::axpr::CoreExpr;
using Lambda = ap::axpr::Lambda<CoreExpr>;

using DrrValue = ap::drr::Value;
using DrrNode = ap::drr::Node<DrrValue>;
using DrrCtx = ap::drr::DrrCtx<DrrValue, DrrNode>;

}  // namespace

adt::Result<DrrCtx> ApDrrHelper::Interpret(const Lambda& lambda,
                                           const std::string& drr_pass_name) {
  ap::axpr::CpsExprInterpreter<DrrValue> interpreter{};
  ADT_LET_CONST_REF(drr_ctx_val, interpreter.Interpret(lambda, {}));
  ADT_LET_CONST_REF(drr_ctx, drr_ctx_val.template TryGet<DrrCtx>())
      << adt::errors::TypeError{
             std::string() +
             "drr function should return a 'DrrCtx' object but '" +
             ap::axpr::GetTypeName(drr_ctx_val) + "' were given."};
  return drr_ctx;
}

}  // namespace cinn::dialect::ir
