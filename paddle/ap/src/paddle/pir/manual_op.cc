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
    LoadFromRegisterOp::attributes_name[LoadFromRegisterOp::attributes_num] = {
        "unique_name"};

void LoadFromRegisterOp::Build(pir::Builder& builder,
                               pir::OperationArgument& argument,
                               pir::Value x,
                               const std::string& name) {
  argument.inputs = {x};
  argument.output_types = {x.type()};
  argument.AddAttribute(
      "unique_name", pir::StrAttribute::get(pir::IrContext::Instance(), name));
}

bool LoadFromRegisterOp::InferSymbolicShape(
    pir::InferSymbolicShapeContext* infer_context) {
  infer_context->SetShapeOrDataForValue(
      result(0), infer_context->GetShapeOrDataForValue(operand_source(0)));
  return true;
}

const char*
    StoreToRegisterOp::attributes_name[StoreToRegisterOp::attributes_num] = {
        "unique_name"};

void StoreToRegisterOp::Build(pir::Builder& builder,
                              pir::OperationArgument& argument,
                              pir::Value x,
                              const std::string& name) {
  argument.inputs = {x};
  argument.output_types = {x.type()};
  argument.AddAttribute(
      "unique_name", pir::StrAttribute::get(pir::IrContext::Instance(), name));
}

bool StoreToRegisterOp::InferSymbolicShape(
    pir::InferSymbolicShapeContext* infer_context) {
  infer_context->SetShapeOrDataForValue(
      result(0), infer_context->GetShapeOrDataForValue(operand_source(0)));
  return true;
}

const char*
    LoadFromGlobalOp::attributes_name[LoadFromGlobalOp::attributes_num] = {
        "index_func_unique_id"};

void LoadFromGlobalOp::Build(pir::Builder& builder,
                             pir::OperationArgument& argument,
                             pir::Value input,
                             const std::string& index_func_unique_id) {
  argument.inputs = {input};
  argument.output_types = {input.type()};
  argument.AddAttribute(
      "index_func_unique_id",
      pir::StrAttribute::get(pir::IrContext::Instance(), index_func_unique_id));
}

bool LoadFromGlobalOp::InferSymbolicShape(
    pir::InferSymbolicShapeContext* infer_context) {
  infer_context->SetShapeOrDataForValue(
      result(0), infer_context->GetShapeOrDataForValue(operand_source(0)));
  return true;
}

const char* StoreToGlobalOp::attributes_name[StoreToGlobalOp::attributes_num] =
    {"index_func_unique_id"};

void StoreToGlobalOp::Build(pir::Builder& builder,
                            pir::OperationArgument& argument,
                            pir::Value input,
                            const std::string& index_func_unique_id) {
  argument.inputs = {input};
  argument.output_types = {input.type()};
  argument.AddAttribute(
      "index_func_unique_id",
      pir::StrAttribute::get(pir::IrContext::Instance(), index_func_unique_id));
}

bool StoreToGlobalOp::InferSymbolicShape(
    pir::InferSymbolicShapeContext* infer_context) {
  infer_context->SetShapeOrDataForValue(
      result(0), infer_context->GetShapeOrDataForValue(operand_source(0)));
  return true;
}

}  // namespace ap::dialect

IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::UpSpiderOp);
IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::DownSpiderOp);
IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::LoadFromRegisterOp);
IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::StoreToRegisterOp);
IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::LoadFromGlobalOp);
IR_DEFINE_EXPLICIT_TYPE_ID(ap::dialect::StoreToGlobalOp);
