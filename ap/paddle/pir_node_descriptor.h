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
        [](const NativeIrValue& ir_value) -> std::string {
          return std::to_string(
              reinterpret_cast<int64_t>(ir_value.value.impl()));
        },
        [](const PackedIrValue& ir_value) -> std::string {
          pir::Operation* op = ir_value.fusion_op;
          return std::to_string(reinterpret_cast<int64_t>(op)) +
                 std::to_string(ir_value.is_output ? 1 : 0);
        },
        [](const NativeIrOpOperand& ir_op_operand) -> std::string {
          return std::to_string(reinterpret_cast<int64_t>(
                     ir_op_operand.op_operand.owner())) +
                 std::to_string(ir_op_operand.op_operand.index());
        },
        [](const PackedIrOpOperand& ir_op_operand) -> std::string {
          pir::Operation* op = ir_op_operand.fusion_op;
          return std::to_string(reinterpret_cast<int64_t>(op)) +
                 std::to_string(ir_op_operand.free_tensor_index);
        },
        [](const NativeIrOp& ir_op) -> std::string {
          return std::to_string(reinterpret_cast<int64_t>(ir_op.op));
        },
        [](const PackedIrOp& ir_op) -> std::string {
          pir::Operation* op = ir_op.fusion_op;
          return std::to_string(reinterpret_cast<int64_t>(op));
        },
        [](const NativeIrOpResult& ir_op_result) -> std::string {
          pir::Operation* op = ir_op_result.op_result.owner();
          return std::to_string(reinterpret_cast<int64_t>(op)) +
                 std::to_string(ir_op_result.op_result.index());
        },
        [](const PackedIrOpResult& ir_op_result) -> std::string {
          pir::Operation* op = ir_op_result.op_result.owner();
          return std::to_string(reinterpret_cast<int64_t>(op)) +
                 std::to_string(ir_op_result.op_result.index());
        });
  }
};

}  // namespace ap::paddle

namespace ap::graph {

template <>
struct NodeDescriptor<ap::paddle::PirNode>
    : public ap::paddle::PirNodeDescriptor {};

}  // namespace ap::graph
