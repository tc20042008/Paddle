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

#include "ap/drr/drr_value.h"
#include "ap/drr/node.h"
#include "ap/graph/node.h"
#include "ap/graph/node_descriptor.h"

namespace ap::drr {

template <typename ValueT>
struct DrrNodeDescriptor {
  using DrrNode = drr::Node<ValueT>;

  using DrrNativeIrValue = ap::drr::NativeIrValue<DrrNode>;
  using DrrPackedIrValue = ap::drr::PackedIrValue<DrrNode>;
  using DrrNativeIrOp = ap::drr::NativeIrOp<ValueT, DrrNode>;
  using DrrPackedIrOp = ap::drr::PackedIrOp<ValueT, DrrNode>;
  using DrrNativeIrOpOperand = ap::drr::NativeIrOpOperand<DrrNode>;
  using DrrPackedIrOpOperand = ap::drr::PackedIrOpOperand<DrrNode>;
  using DrrNativeIrOpResult = ap::drr::NativeIrOpResult<DrrNode>;
  using DrrPackedIrOpResult = ap::drr::PackedIrOpResult<DrrNode>;

  std::string DebugId(const graph::Node<DrrNode>& node) {
    const auto& opt_drr_node = node.Get();
    if (opt_drr_node.HasError()) {
      return std::to_string(node.node_id().value());
    }
    const auto& drr_node = opt_drr_node.GetOkValue();
    return drr_node.Match(
        [&](const DrrNativeIrValue& ir_value) -> std::string {
          return ir_value->name;
        },
        [&](const DrrPackedIrValue& ir_value) -> std::string {
          return ir_value->name;
        },
        [&](const DrrNativeIrOp& ir_op) -> std::string {
          return ir_op->op_declare->op_name + "[" + ir_op->name + "]";
        },
        [&](const DrrPackedIrOp& ir_op) -> std::string {
          return ir_op->op_declare->op_name + "[" + ir_op->name + "]";
        },
        [&](const DrrNativeIrOpOperand& ir_op_operand) -> std::string {
          return EdgeDebugId(node);
        },
        [&](const DrrPackedIrOpOperand& ir_op_operand) -> std::string {
          return EdgeDebugId(node);
        },
        [&](const DrrNativeIrOpResult& ir_op_result) -> std::string {
          return EdgeDebugId(node);
        },
        [&](const DrrPackedIrOpResult& ir_op_result) -> std::string {
          return EdgeDebugId(node);
        });
  }

  std::string EdgeDebugId(const graph::Node<DrrNode>& node) {
    const auto& opt_src_and_dst = GetSrcAndDst(node);
    if (!opt_src_and_dst.has_value()) {
      return std::string("invalid_edge_") +
             std::to_string(node.node_id().value());
    }
    const auto& [src, dst] = opt_src_and_dst.value();
    return DebugId(src) + "->" + DebugId(dst);
  }

  struct SrcAndDst {
    graph::Node<DrrNode> src;
    graph::Node<DrrNode> dst;
  };

  std::optional<SrcAndDst> GetSrcAndDst(const graph::Node<DrrNode>& node) {
    const auto& opt_src_and_dst = TryGetSrcAndDst(node);
    if (opt_src_and_dst.HasError()) {
      return std::nullopt;
    }
    return opt_src_and_dst.GetOkValue();
  }

  adt::Result<SrcAndDst> TryGetSrcAndDst(const graph::Node<DrrNode>& node) {
    ADT_LET_CONST_REF(upstreams, node.UpstreamNodes());
    ADT_LET_CONST_REF(downstreams, node.DownstreamNodes());
    ADT_LET_CONST_REF(src, upstreams.Sole());
    ADT_LET_CONST_REF(dst, downstreams.Sole());
    return SrcAndDst{src, dst};
  }
};

}  // namespace ap::drr

namespace ap::graph {

template <typename ValueT>
struct NodeDescriptor<graph::Node<drr::Node<ValueT>>>
    : public drr::DrrNodeDescriptor<ValueT> {};

}  // namespace ap::graph
