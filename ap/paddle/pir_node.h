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

#pragma once

#include <functional>
#include "ap/adt/adt.h"
#include "ap/axpr/builtin_class_instance.h"
#include "ap/axpr/data_type.h"
#include "ap/axpr/data_type_util.h"
#include "ap/axpr/type.h"
#include "ap/graph/node_cstr.h"
#include "ap/ir_match/ref_match_ctx.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/pir/include/core/op_operand.h"
#include "paddle/pir/include/core/op_result.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

namespace ap::paddle {

template <typename ValueT>
const axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>&
GetNativeIrValueClass();

template <typename ValueT>
const axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>&
GetPackedIrValueClass();

template <typename ValueT>
const axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>& GetRefIrValueClass();

template <typename ValueT>
const axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>& GetNativeIrOpClass();

template <typename ValueT>
const axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>& GetPackedIrOpClass();

template <typename ValueT>
const axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>& GetRefIrOpClass();

struct NativeIrValue {
  pir::Value value;

  template <typename ValueT>
  static const axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>&
  GetBuiltinClass() {
    return GetNativeIrValueClass<ValueT>();
  }

  std::size_t GetHashValue() const { return std::hash<pir::Value>()(value); }

  bool operator==(const NativeIrValue& other) const {
    return this->value == other.value;
  }

  graph::NativeIrValueCstr node_cstr() const {
    return graph::NativeIrValueCstr{};
  }

  adt::Result<axpr::DataType> GetDataType() const {
    ADT_LET_CONST_REF(type, GetPhiDataType());
    return ap::axpr::GetDataTypeFromPhiDataType(type);
  }

  adt::Result<const std::vector<symbol::DimExpr>*> GetShapeDimExprsPtr() const {
    auto* op = value.defining_op();
    ADT_CHECK(op != nullptr);
    auto* program = op->GetParentProgram();
    auto& shape_analysis = ::pir::ShapeAnalysisManager::Instance().Get(program);
    const auto& shape_or_data = shape_analysis.GetShapeOrDataForValue(value);
    using RetT = adt::Result<const std::vector<symbol::DimExpr>*>;
    return shape_or_data.Match(
        [&](const symbol::TensorShapeOrDataDimExprs& impl) -> RetT {
          return &impl.shape();
        },
        [&](const auto&) -> RetT {
          return adt::errors::TypeError{
              "GetShapeDimExprsPtr only support TensorShapeOrDataDimExprs."};
        });
  }

 private:
  adt::Result<phi::DataType> GetPhiDataType() const {
    ADT_LET_CONST_REF(type, GetPirDataType());
    try {
      return ::paddle::dialect::TransToPhiDataType(type);
    } catch (const std::exception& e) {
      return adt::errors::TypeError{
          "failed to cast from pir data type to phi data type."};
    }
  }

  adt::Result<pir::Type> GetPirDataType() const {
    if (!this->value.type().isa<pir::DenseTensorType>()) {
      return adt::errors::NotImplementedError{
          "pir value must be of DenseTensorType"};
    }
    const auto dense_tensor_type =
        this->value.type().dyn_cast<pir::DenseTensorType>();
    return dense_tensor_type.dtype();
  }
};

struct PackedIrValue {
  cinn::dialect::FusionOp fusion_op;
  bool is_output;

  template <typename ValueT>
  static const axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>&
  GetBuiltinClass() {
    return GetPackedIrValueClass<ValueT>();
  }

  std::size_t GetHashValue() const {
    return std::hash<pir::Operation*>()(
               static_cast<pir::Operation*>(fusion_op)) ^
           is_output;
  }

  bool operator==(const PackedIrValue& other) const {
    return this->fusion_op == other.fusion_op &&
           this->is_output == other.is_output;
  }

  graph::PackedIrValueCstr node_cstr() const {
    return graph::PackedIrValueCstr{};
  }
};

struct NativeIrOpOperand {
  pir::OpOperand op_operand;

  std::size_t GetHashValue() const {
    return std::hash<pir::OpOperand>()(op_operand);
  }

  bool operator==(const NativeIrOpOperand& other) const {
    return this->op_operand == other.op_operand;
  }

  graph::NativeIrOpOperandCstr node_cstr() const {
    return graph::NativeIrOpOperandCstr{this->op_operand.index()};
  }
};

struct PackedIrOpOperand {
  cinn::dialect::FusionOp fusion_op;
  std::size_t free_tensor_index;

  std::size_t GetHashValue() const {
    return std::hash<pir::Operation*>()(
               static_cast<pir::Operation*>(fusion_op)) ^
           free_tensor_index;
  }

  bool operator==(const PackedIrOpOperand& other) const {
    return this->fusion_op == other.fusion_op &&
           this->free_tensor_index == other.free_tensor_index;
  }

  graph::PackedIrOpOperandCstr node_cstr() const {
    return graph::PackedIrOpOperandCstr{};
  }
};

struct NativeIrOp {
  pir::Operation* op;

  template <typename ValueT>
  static const axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>&
  GetBuiltinClass() {
    return GetNativeIrOpClass<ValueT>();
  }

  std::size_t GetHashValue() const { return std::hash<pir::Operation*>()(op); }

  bool operator==(const NativeIrOp& other) const {
    return this->op == other.op;
  }

  graph::NativeIrOpCstr node_cstr() const {
    return graph::NativeIrOpCstr{this->op->name()};
  }
};

struct PackedIrOp {
  cinn::dialect::FusionOp fusion_op;

  template <typename ValueT>
  static const axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>&
  GetBuiltinClass() {
    return GetPackedIrOpClass<ValueT>();
  }

  std::size_t GetHashValue() const {
    return std::hash<pir::Operation*>()(
        static_cast<pir::Operation*>(fusion_op));
  }

  bool operator==(const PackedIrOp& other) const {
    return this->fusion_op == other.fusion_op;
  }

  graph::PackedIrOpCstr node_cstr() const {
    return graph::PackedIrOpCstr{"ap_trivial_fusion_op"};
  }
};

struct NativeIrOpResult {
  pir::OpResult op_result;

  std::size_t GetHashValue() const {
    return std::hash<pir::OpResult>()(op_result);
  }

  bool operator==(const NativeIrOpResult& other) const {
    return this->op_result == other.op_result;
  }

  graph::NativeIrOpResultCstr node_cstr() const {
    return graph::NativeIrOpResultCstr{this->op_result.index()};
  }
};

struct PackedIrOpResult {
  pir::OpResult op_result;

  std::size_t GetHashValue() const {
    return std::hash<pir::OpResult>()(op_result);
  }

  bool operator==(const PackedIrOpResult& other) const {
    return this->op_result == other.op_result;
  }

  graph::PackedIrOpResultCstr node_cstr() const {
    return graph::PackedIrOpResultCstr{};
  }
};

}  // namespace ap::paddle

namespace std {

template <>
struct hash<ap::paddle::NativeIrValue> {
  std::size_t operator()(const ap::paddle::NativeIrValue& node) const {
    return node.GetHashValue();
  }
};

template <>
struct hash<ap::paddle::NativeIrOpOperand> {
  std::size_t operator()(const ap::paddle::NativeIrOpOperand& node) const {
    return node.GetHashValue();
  }
};

}  // namespace std

namespace ap::paddle {

using RefNodeInfo = ir_match::RefNodeInfo<NativeIrValue, NativeIrOpOperand>;

struct RefIrValue {
  RefNodeInfo ref_node_info;

  template <typename ValueT>
  static const axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>&
  GetBuiltinClass() {
    return GetRefIrValueClass<ValueT>();
  }

  std::size_t GetHashValue() const {
    return std::hash<RefNodeInfo>()(ref_node_info);
  }

  bool operator==(const RefIrValue& other) const {
    return this->ref_node_info == other.ref_node_info;
  }

  adt::Result<NativeIrValue> GetOwnerNativeIrValue() const {
    return this->ref_node_info->ir_value;
  }

  graph::RefIrValueCstr node_cstr() const { return graph::RefIrValueCstr{}; }
};

struct RefIrOpOperand {
  RefNodeInfo ref_node_info;

  std::size_t GetHashValue() const {
    return std::hash<RefNodeInfo>()(ref_node_info);
  }

  bool operator==(const RefIrOpOperand& other) const {
    return this->ref_node_info == other.ref_node_info;
  }

  graph::RefIrOpOperandCstr node_cstr() const {
    return graph::RefIrOpOperandCstr{};
  }
};

struct RefIrOp {
  RefNodeInfo ref_node_info;

  template <typename ValueT>
  static const axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>&
  GetBuiltinClass() {
    return GetRefIrOpClass<ValueT>();
  }

  std::size_t GetHashValue() const {
    return std::hash<RefNodeInfo>()(ref_node_info);
  }

  bool operator==(const RefIrOp& other) const {
    return this->ref_node_info == other.ref_node_info;
  }

  graph::RefIrOpCstr node_cstr() const { return graph::RefIrOpCstr{}; }
};

struct RefIrOpResult {
  RefNodeInfo ref_node_info;

  std::size_t GetHashValue() const {
    return std::hash<RefNodeInfo>()(ref_node_info);
  }

  bool operator==(const RefIrOpResult& other) const {
    return this->ref_node_info == other.ref_node_info;
  }

  graph::RefIrOpResultCstr node_cstr() const {
    return graph::RefIrOpResultCstr{};
  }
};

using PirNodeImpl = std::variant<NativeIrValue,
                                 PackedIrValue,
                                 NativeIrOpOperand,
                                 PackedIrOpOperand,
                                 RefIrOpOperand,
                                 NativeIrOp,
                                 PackedIrOp,
                                 NativeIrOpResult,
                                 PackedIrOpResult,
                                 RefIrValue,
                                 RefIrOp,
                                 RefIrOpResult>;

struct PirNode : public PirNodeImpl {
  using PirNodeImpl::PirNodeImpl;
  DEFINE_ADT_VARIANT_METHODS(PirNodeImpl);

  using dim_expr_type = ::symbol::DimExpr;
  using native_op_type = NativeIrOp;
  using packed_op_type = PackedIrOp;
  using ref_op_type = RefIrOp;
  using native_value_type = NativeIrValue;
  using packed_value_type = PackedIrValue;
  using ref_value_type = RefIrValue;
  using native_op_operand_type = NativeIrOpOperand;

  std::size_t GetHashValue() const {
    return Match([](const auto& impl) { return impl.GetHashValue(); });
  }

  graph::NodeCstr node_cstr() const {
    return Match(
        [](const auto& impl) -> graph::NodeCstr { return impl.node_cstr(); });
  }

  static adt::Result<std::string> GetOpNameFromDrrPackedOpName(
      const std::string& drr_packed_op_name) {
    if (drr_packed_op_name == "ap_trivial_fusion_op") {
      return "cinn_op.fusion";
    }
    return adt::errors::KeyError{
        std::string() + "no pir op name matched to drr packed op name: '" +
        drr_packed_op_name + "'"};
  }
};

}  // namespace ap::paddle

namespace std {

template <>
struct hash<ap::paddle::PirNode> {
  std::size_t operator()(const ap::paddle::PirNode& node) const {
    return node.GetHashValue();
  }
};

}  // namespace std
