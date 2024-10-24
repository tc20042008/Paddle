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

#include "ap/adt/adt.h"
#include "ap/graph/graph_descriptor.h"
#include "ap/graph/node.h"
#include "ap/paddle/pir_node.h"
#include "ap/paddle/pir_util.h"

namespace ap::paddle {

struct PirGraphDescriptor {
  using NodeT = PirNode;

  NodeT CastToIrOpResult(const pir::OpResult& op_result) const {
    if (op_result.owner()->isa<cinn::dialect::FusionOp>()) {
      return PackedIrOpResult{op_result};
    } else {
      return NativeIrOpResult{op_result};
    }
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitUpstreamNodes(const NodeT& node,
                                          const DoEachT& DoEach) const {
    return node.Match(
        [&](const NativeIrValue& impl) -> adt::Result<adt::Ok> {
          if (pir::OpResult::classof(impl.value)) {
            return DoEach(
                CastToIrOpResult(pir::OpResult::dyn_cast_from(impl.value)));
          }
          return adt::Ok{};
        },
        [&](const PackedIrValue& impl) -> adt::Result<adt::Ok> {
          // TODO(tianchao): support the following case:
          // o.trivial_op0([*t.inputs], [t.op0_output, *t.op0_output1])
          // o.trivial_op1([*.t.op0_output1], [t.op1_output])
          return adt::errors::NotImplementedError{
              "PirGraphDescriptor::VisitUpstreamNodes does not support "
              "PackedIrValue"};
        },
        [&](const NativeIrOpOperand& impl) -> adt::Result<adt::Ok> {
          NativeIrValue ir_value{impl.op_operand.source()};
          return DoEach(ir_value);
        },
        [&](const PackedIrOpOperand& impl) -> adt::Result<adt::Ok> {
          const auto& inputs = GetFusionOpInputValues(impl.fusion_op);
          ADT_CHECK(impl.free_tensor_index >= 0);
          ADT_CHECK(impl.free_tensor_index < inputs.size());
          NativeIrValue ir_value{inputs.at(impl.free_tensor_index)};
          return DoEach(ir_value);
        },
        [&](const NativeIrOp& impl) -> adt::Result<adt::Ok> {
          for (int i = 0; i < impl.op->num_operands(); ++i) {
            NativeIrOpOperand ir_op_operand{impl.op->operand(i)};
            ADT_RETURN_IF_ERR(DoEach(ir_op_operand));
          }
          return adt::Ok{};
        },
        [&](const PackedIrOp& impl) -> adt::Result<adt::Ok> {
          const auto& inputs = GetFusionOpInputValues(impl.fusion_op);
          for (int i = 0; i < inputs.size(); ++i) {
            PackedIrOpOperand ir_op_operand{impl.fusion_op, i};
            ADT_RETURN_IF_ERR(DoEach(ir_op_operand));
          }
          return adt::Ok{};
        },
        [&](const NativeIrOpResult& impl) -> adt::Result<adt::Ok> {
          NativeIrOp ir_op{impl.op_result.defining_op()};
          return DoEach(ir_op);
        },
        [&](const PackedIrOpResult& impl) -> adt::Result<adt::Ok> {
          auto* op = impl.op_result.defining_op();
          ADT_CHECK(op->isa<cinn::dialect::FusionOp>());
          PackedIrOp ir_op{op->dyn_cast<cinn::dialect::FusionOp>()};
          return DoEach(ir_op);
        });
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitDownstreamNodes(const NodeT& node,
                                            const DoEachT& DoEach) const {
    return node.Match(
        [&](const NativeIrValue& impl) -> adt::Result<adt::Ok> {
          for (auto iter = impl.value.use_begin(); iter != impl.value.use_end();
               ++iter) {
            auto* user_parent_block = iter->owner()->GetParent();
            ADT_CHECK(user_parent_block != nullptr);
            auto* user_parent_op = user_parent_block->GetParentOp();
            if (user_parent_op->isa<cinn::dialect::FusionOp>()) {
              auto fusion_op =
                  user_parent_op->dyn_cast<cinn::dialect::FusionOp>();
              const auto& user_op_inputs = GetFusionOpInputValues(fusion_op);
              for (int i = 0; i < user_op_inputs.size(); ++i) {
                if (user_op_inputs.at(i) == impl.value) {
                  PackedIrOpOperand ir_op_operand{fusion_op, i};
                  ADT_RETURN_IF_ERR(DoEach(ir_op_operand));
                }
              }
            } else {
              pir::OpOperand op_operand = *iter;
              NativeIrOpOperand ir_op_operand{op_operand};
              ADT_RETURN_IF_ERR(DoEach(ir_op_operand));
            }
          }
          return adt::Ok{};
        },
        [&](const PackedIrValue& impl) -> adt::Result<adt::Ok> {
          // TODO(tianchao): support the following case:
          // o.trivial_op0([*t.inputs], [t.op0_output, *t.op0_output1])
          // o.trivial_op1([*.t.op0_output1], [t.op1_output])
          return adt::Ok{};
        },
        [&](const NativeIrOpOperand& impl) -> adt::Result<adt::Ok> {
          NativeIrOp ir_op{impl.op_operand.owner()};
          return DoEach(ir_op);
        },
        [&](const PackedIrOpOperand& impl) -> adt::Result<adt::Ok> {
          PackedIrOp ir_op{impl.fusion_op};
          return DoEach(ir_op);
        },
        [&](const NativeIrOp& impl) -> adt::Result<adt::Ok> {
          for (int i = 0; i < impl.op->num_results(); ++i) {
            const auto& value = impl.op->result(i);
            ADT_CHECK(pir::OpResult::classof(value));
            NativeIrOpResult ir_op_result{pir::OpResult::dyn_cast_from(value)};
            ADT_RETURN_IF_ERR(DoEach(ir_op_result));
          }
          return adt::Ok{};
        },
        [&](const PackedIrOp& impl) -> adt::Result<adt::Ok> {
          for (int i = 0; i < impl.fusion_op->num_results(); ++i) {
            const auto& value = impl.fusion_op->result(i);
            ADT_CHECK(pir::OpResult::classof(value));
            PackedIrOpResult ir_op_result{pir::OpResult::dyn_cast_from(value)};
            ADT_RETURN_IF_ERR(DoEach(ir_op_result));
          }
          return adt::Ok{};
        },
        [&](const NativeIrOpResult& impl) -> adt::Result<adt::Ok> {
          pir::Value value = impl.op_result;
          NativeIrValue ir_value{value};
          return DoEach(ir_value);
        },
        [&](const PackedIrOpResult& impl) -> adt::Result<adt::Ok> {
          pir::Value value = impl.op_result;
          NativeIrValue ir_value{value};
          return DoEach(ir_value);
        });
  }

  adt::Result<graph::NodeCstr> GetNodeConstraint(const NodeT& node) const {
    return node.node_cstr();
  }

  adt::Result<bool> IgnoredNode(const NodeT& node) const {
    return node.Match(
        [](const PackedIrValue&) -> adt::Result<bool> { return true; },
        [](const auto&) -> adt::Result<bool> { return false; });
  }

  adt::Result<bool> IsOpNode(const NodeT& node) const {
    return node.Match([&](const NativeIrOp&) -> bool { return true; },
                      [&](const PackedIrOp&) -> bool { return true; },
                      [&](const auto&) -> bool { return false; });
  }

  adt::Result<bool> Satisfy(const NodeT& node,
                            const graph::NodeCstr& node_cstr) const {
    return node.node_cstr() == node_cstr;
  }

  const std::vector<pir::Value>& GetFusionOpInputValues(
      cinn::dialect::FusionOp fusion_op) const {
    auto iter = fusion_op2input_values_.find(fusion_op);
    if (iter == fusion_op2input_values_.end()) {
      iter =
          fusion_op2input_values_
              .emplace(fusion_op, ap::paddle::GetUsedExternalValue(*fusion_op))
              .first;
    }
    return iter->second;
  }

 private:
  mutable std::unordered_map<pir::Operation*, std::vector<pir::Value>>
      fusion_op2input_values_;
};

}  // namespace ap::paddle

namespace ap::graph {

template <>
struct GraphDescriptor<ap::paddle::PirNode>
    : public ap::paddle::PirGraphDescriptor {};

}  // namespace ap::graph
