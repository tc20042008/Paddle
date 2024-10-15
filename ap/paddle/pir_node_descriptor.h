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

#include "ap/graph/node_descriptor.h"
#include "ap/paddle/pir_node.h"

namespace ap::paddle {

struct PirNodeDescriptor {
  std::string DebugId(const PirNode& node) {
    return node.Match(
        [&](const NativeIrValue& ir_value) -> std::string {
          if (ir_value.value.defining_op() == nullptr) {
            return std::to_string(
                reinterpret_cast<int64_t>(ir_value.value.impl()));
          } else {
            const auto* op = ir_value.value.defining_op();
            const auto& op_debug_id = GetOpDebugId(op);
            for (int i = 0; i < op->num_results(); ++i) {
              if (op->result(i) == ir_value.value) {
                return op_debug_id + "_out_" + std::to_string(i);
              }
            }
            return op_debug_id + "_error_output";
          }
        },
        [&](const PackedIrValue& ir_value) -> std::string {
          pir::Operation* op = ir_value.fusion_op;
          const auto& op_debug_id = GetOpDebugId(op);
          if (ir_value.is_output) {
            return op_debug_id + "_packed_out";
          } else {
            return op_debug_id + "_packed_in";
          }
        },
        [&](const NativeIrOpOperand& ir_op_operand) -> std::string {
          const auto& operand = ir_op_operand.op_operand;
          const auto& op_debug_id = GetOpDebugId(operand.owner());
          return op_debug_id + "_operand_" + std::to_string(operand.index());
        },
        [&](const PackedIrOpOperand& ir_op_operand) -> std::string {
          pir::Operation* op = ir_op_operand.fusion_op;
          const auto& op_debug_id = GetOpDebugId(op);
          std::size_t index = ir_op_operand.free_tensor_index;
          return op_debug_id + "_packed_operand_" + std::to_string(index);
        },
        [&](const NativeIrOp& ir_op) -> std::string {
          return GetOpDebugId(ir_op.op);
        },
        [&](const PackedIrOp& ir_op) -> std::string {
          pir::Operation* op = ir_op.fusion_op;
          return GetOpDebugId(op);
        },
        [&](const NativeIrOpResult& ir_op_result) -> std::string {
          pir::Operation* op = ir_op_result.op_result.owner();
          const auto& op_debug_id = GetOpDebugId(op);
          std::size_t index = ir_op_result.op_result.index();
          return op_debug_id + "_result_" + std::to_string(index);
        },
        [&](const PackedIrOpResult& ir_op_result) -> std::string {
          pir::Operation* op = ir_op_result.op_result.owner();
          const auto& op_debug_id = GetOpDebugId(op);
          std::size_t index = ir_op_result.op_result.index();
          return op_debug_id + "_packed_result_" + std::to_string(index);
        });
  }

  std::string GetOpDebugId(const pir::Operation* op) const {
    return op->name() + "_" + std::to_string(op->id());
  }
};

}  // namespace ap::paddle

namespace ap::graph {

template <>
struct NodeDescriptor<ap::paddle::PirNode>
    : public ap::paddle::PirNodeDescriptor {};

}  // namespace ap::graph
