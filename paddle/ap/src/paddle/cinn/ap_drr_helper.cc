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

#include "paddle/ap/include/paddle/cinn/ap_drr_helper.h"
#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/interpreter.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/drr/builtin_frame_util.h"
#include "paddle/ap/include/drr/drr_graph_descriptor.h"
#include "paddle/ap/include/drr/drr_interpreter.h"
#include "paddle/ap/include/drr/drr_node_descriptor.h"
#include "paddle/ap/include/drr/value_method_class.h"
#include "paddle/ap/include/paddle/pir/pir_method_class.h"

namespace cinn::dialect::ir {

namespace adt = ap::adt;

namespace {

using Function = ap::axpr::Function<ap::axpr::SerializableValue>;

using DrrNode = ap::drr::Node;
using DrrCtx = ap::drr::DrrCtx;

}  // namespace

adt::Result<DrrCtx> ApDrrHelper::Interpret(const Function& lambda,
                                           const std::string& drr_pass_name) {
  ap::drr::DrrInterpreter drr_interpreter{};
  return drr_interpreter.Interpret(
      ap::paddle::GetPirClass(), lambda, drr_pass_name);
}

adt::Result<DrrCtx> ApDrrHelper::Interpret(
    const ap::axpr::ClassAttrs<ap::axpr::SerializableValue>& cls) {
  ap::drr::DrrInterpreter drr_interpreter{};
  return drr_interpreter.Interpret(ap::paddle::GetPirClass(), cls);
}

}  // namespace cinn::dialect::ir
