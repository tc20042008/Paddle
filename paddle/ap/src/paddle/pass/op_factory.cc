// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/src/paddle/pass/op_factory.h"
#include "paddle/ap/include/paddle/pir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

namespace ap::paddle {

namespace {

adt::Result<pir::Operation*> ConstructPdOpSum(
    pir::Builder* builder,
    const std::vector<pir::Value>& inputs,
    pir::AttributeMap attrs) {
  ADT_CHECK(inputs.size() == 1);
  attrs["dtype"] = ::paddle::dialect::DataTypeAttribute::get(
      pir::IrContext::Instance(), phi::DataType::UNDEFINED);
  auto op = builder->Build<::paddle::dialect::SumOp>(inputs.at(0), attrs);
  return op;
}

adt::Result<pir::Operation*> ConstructUpSpiderOp(
    pir::Builder* builder,
    const std::vector<pir::Value>& inputs,
    const pir::AttributeMap& attrs) {
  ADT_CHECK(inputs.size() == 2) << adt::errors::TypeError{
      std::string() + "'ap_op.up_spider' op takes 2 arguments, but " +
      std::to_string(inputs.size()) + " were given"};
  auto op = builder->Build<ap::dialect::UpSpiderOp>(inputs.at(0), inputs.at(1));
  return op;
}

adt::Result<pir::Operation*> ConstructYieldOp(
    pir::Builder* builder,
    const std::vector<pir::Value>& inputs,
    const pir::AttributeMap& attrs) {
  auto op = builder->Build<pir::YieldOp>(inputs);
  return op;
}

adt::Result<pir::Operation*> ConstructShadowOutputOp(
    pir::Builder* builder,
    const std::vector<pir::Value>& inputs,
    const pir::AttributeMap& attrs) {
  ADT_CHECK(inputs.size() == 1);
  const auto& iter = attrs.find("output_name");
  ADT_CHECK(iter != attrs.end());
  ADT_CHECK(iter->second.isa<pir::StrAttribute>());
  const std::string& output_name =
      iter->second.dyn_cast<pir::StrAttribute>().AsString();
  auto op = builder->Build<pir::ShadowOutputOp>(inputs.at(0), output_name);
  return op;
}

adt::Result<pir::Operation*> ConstructDownSpiderOp(
    pir::Builder* builder,
    const std::vector<pir::Value>& inputs,
    const pir::AttributeMap& attrs) {
  ADT_CHECK(inputs.size() == 1);
  auto op = builder->Build<ap::dialect::DownSpiderOp>(inputs.at(0));
  return op;
}

}  // namespace

adt::Result<std::optional<pir::Operation*>> CreateOperation(
    pir::Builder* builder,
    const std::string& op_name,
    const std::vector<pir::Value>& inputs,
    const pir::AttributeMap& attrs) {
  if (op_name == "pd_op.sum") {
    ADT_LET_CONST_REF(ret, ConstructPdOpSum(builder, inputs, attrs));
    return ret;
  }
  if (op_name == "cf.yield") {
    ADT_LET_CONST_REF(ret, ConstructYieldOp(builder, inputs, attrs));
    return ret;
  }
  if (op_name == "builtin.shadow_output") {
    ADT_LET_CONST_REF(ret, ConstructShadowOutputOp(builder, inputs, attrs));
    return ret;
  }
  if (op_name == "ap_op.up_spider") {
    ADT_LET_CONST_REF(ret, ConstructUpSpiderOp(builder, inputs, attrs));
    return ret;
  }
  if (op_name == "ap_op.down_spider") {
    ADT_LET_CONST_REF(ret, ConstructDownSpiderOp(builder, inputs, attrs));
    return ret;
  }
  return std::nullopt;
}

}  // namespace ap::paddle
