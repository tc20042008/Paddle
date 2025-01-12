// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <glog/logging.h>

#include "paddle/ap/include/paddle/pir/manual_op.h"
#include "paddle/common/enforce.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"

namespace ap::dialect {

void IdUpSpider::Build(pir::Builder& builder,             // NOLINT
                       pir::OperationArgument& argument,  // NOLINT
                       pir::Value lhs,
                       pir::Value rhs) {
  argument.AddInput(lhs);
  argument.AddInput(rhs);
}

void IdDownSpider::Build(pir::Builder& builder,
                         pir::OperationArgument& argument,
                         pir::Value x,
                         pir::Type output_type) {
  argument.inputs = {x};
  argument.output_types = {output_type};
}

bool IdDownSpider::InferSymbolicShape(
    pir::InferSymbolicShapeContext* infer_context) {
  infer_context->SetShapeOrDataForValue(
      result(0), infer_context->GetShapeOrDataForValue(operand_source(0)));
  return true;
}

}  // namespace ap::dialect

IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::IdUpSpider)
IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::IdDownSpider)
