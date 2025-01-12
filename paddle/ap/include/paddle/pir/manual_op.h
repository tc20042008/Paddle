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

#pragma once

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_symbolic_shape.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/operation_utils.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

namespace ap::dialect {

class IR_API IdUpSpider : public pir::Op<IdUpSpider,
                                         pir::SideEffectTrait,
                                         pir::ImmutableLayoutTrait> {
 public:
  using Op::Op;
  static const char *name() { return "ap_op.id_up_spider"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value lhs,
                    pir::Value rhs);
  void VerifySig() const {}
};

class IR_API IdDownSpider
    : public pir::Op<IdDownSpider,
                     ::paddle::dialect::InferSymbolicShapeInterface> {
 public:
  using Op::Op;
  static const char *name() { return "ap_op.id_down_spider"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value input,
                    pir::Type output_type);
  void VerifySig() const {}
  bool InferSymbolicShape(pir::InferSymbolicShapeContext *infer_context);
};

}  // namespace ap::dialect

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ap::dialect::IdUpSpider)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ap::dialect::IdDownSpider)
