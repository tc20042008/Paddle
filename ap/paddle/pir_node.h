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
#include "ap/axpr/type.h"
#include "ap/graph/node_cstr.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/pir/include/core/op_operand.h"
#include "paddle/pir/include/core/op_result.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"

namespace ap::paddle {

struct NativeIrValue {
  pir::Value value;

  std::size_t GetHashValue() const { return std::hash<pir::Value>()(value); }

  bool operator==(const NativeIrValue& other) const {
    return this->value == other.value;
  }

  graph::NativeIrValueCstr node_cstr() const {
    return graph::NativeIrValueCstr{};
  }
};

struct PackedIrValue {
  cinn::dialect::FusionOp fusion_op;
  bool is_output;

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

using PirNodeImpl = std::variant<NativeIrValue,
                                 PackedIrValue,
                                 NativeIrOpOperand,
                                 PackedIrOpOperand,
                                 NativeIrOp,
                                 PackedIrOp,
                                 NativeIrOpResult,
                                 PackedIrOpResult>;

struct PirNode : public PirNodeImpl {
  using PirNodeImpl::PirNodeImpl;
  DEFINE_ADT_VARIANT_METHODS(PirNodeImpl);

  using dim_expr_type = ::symbol::DimExpr;
  using native_op_type = NativeIrOp;
  using packed_op_type = PackedIrOp;
  using native_value_type = NativeIrValue;
  using packed_value_type = PackedIrValue;

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

namespace ap::axpr {

template <>
struct TypeImpl<ap::paddle::NativeIrValue> : public std::monostate {
  using value_type = ap::paddle::NativeIrValue;

  const char* Name() const { return "NativeIrValue"; }
};

template <>
struct TypeImpl<ap::paddle::PackedIrValue> : public std::monostate {
  using value_type = ap::paddle::PackedIrValue;

  const char* Name() const { return "PackedIrValue"; }
};

template <>
struct TypeImpl<ap::paddle::NativeIrOp> : public std::monostate {
  using value_type = ap::paddle::NativeIrOp;

  const char* Name() const { return "NativeIrOp"; }
};

template <>
struct TypeImpl<ap::paddle::PackedIrOp> : public std::monostate {
  using value_type = ap::paddle::PackedIrOp;

  const char* Name() const { return "PackedIrOp"; }
};

}  // namespace ap::axpr
