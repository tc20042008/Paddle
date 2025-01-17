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

void UpSpiderOp::Build(pir::Builder& builder,             // NOLINT
                       pir::OperationArgument& argument,  // NOLINT
                       pir::Value lhs,
                       pir::Value rhs) {
  argument.AddInput(lhs);
  argument.AddInput(rhs);
}

void DownSpiderOp::Build(pir::Builder& builder,
                         pir::OperationArgument& argument,
                         pir::Value x) {
  argument.inputs = {x};
  argument.output_types = {x.type()};
}

bool DownSpiderOp::InferSymbolicShape(
    pir::InferSymbolicShapeContext* infer_context) {
  infer_context->SetShapeOrDataForValue(
      result(0), infer_context->GetShapeOrDataForValue(operand_source(0)));
  return true;
}

const char*
    LoadPlaceholderOp::attributes_name[LoadPlaceholderOp::attributes_num] = {
        "unique_name"};

void LoadPlaceholderOp::Build(pir::Builder& builder,
                              pir::OperationArgument& argument,
                              pir::Value x,
                              const std::string& name) {
  argument.inputs = {x};
  argument.output_types = {x.type()};
  argument.AddAttribute(
      "unique_name", pir::StrAttribute::get(pir::IrContext::Instance(), name));
}

bool LoadPlaceholderOp::InferSymbolicShape(
    pir::InferSymbolicShapeContext* infer_context) {
  infer_context->SetShapeOrDataForValue(
      result(0), infer_context->GetShapeOrDataForValue(operand_source(0)));
  return true;
}

const char*
    StorePlaceholderOp::attributes_name[StorePlaceholderOp::attributes_num] = {
        "unique_name"};

void StorePlaceholderOp::Build(pir::Builder& builder,
                               pir::OperationArgument& argument,
                               pir::Value x,
                               const std::string& name) {
  argument.inputs = {x};
  argument.output_types = {x.type()};
  argument.AddAttribute(
      "unique_name", pir::StrAttribute::get(pir::IrContext::Instance(), name));
}

bool StorePlaceholderOp::InferSymbolicShape(
    pir::InferSymbolicShapeContext* infer_context) {
  infer_context->SetShapeOrDataForValue(
      result(0), infer_context->GetShapeOrDataForValue(operand_source(0)));
  return true;
}

const char* LoadOp::attributes_name[LoadOp::attributes_num] = {
    "serialized_index_function"};

void LoadOp::Build(pir::Builder& builder,
                   pir::OperationArgument& argument,
                   pir::Value input,
                   const std::string& serialized_index_function) {
  argument.inputs = {input};
  argument.output_types = {input.type()};
  argument.AddAttribute("serialized_index_function",
                        pir::StrAttribute::get(pir::IrContext::Instance(),
                                               serialized_index_function));
}

bool LoadOp::InferSymbolicShape(pir::InferSymbolicShapeContext* infer_context) {
  infer_context->SetShapeOrDataForValue(
      result(0), infer_context->GetShapeOrDataForValue(operand_source(0)));
  return true;
}

const char* StoreOp::attributes_name[StoreOp::attributes_num] = {
    "serialized_index_function"};

void StoreOp::Build(pir::Builder& builder,
                    pir::OperationArgument& argument,
                    pir::Value input,
                    const std::string& serialized_index_function) {
  argument.inputs = {input};
  argument.output_types = {input.type()};
  argument.AddAttribute("serialized_index_function",
                        pir::StrAttribute::get(pir::IrContext::Instance(),
                                               serialized_index_function));
}

bool StoreOp::InferSymbolicShape(
    pir::InferSymbolicShapeContext* infer_context) {
  infer_context->SetShapeOrDataForValue(
      result(0), infer_context->GetShapeOrDataForValue(operand_source(0)));
  return true;
}

}  // namespace ap::dialect

IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::UpSpiderOp);
IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::DownSpiderOp);
IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::LoadPlaceholderOp);
IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::StorePlaceholderOp);
IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::LoadOp);
IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::StoreOp);
