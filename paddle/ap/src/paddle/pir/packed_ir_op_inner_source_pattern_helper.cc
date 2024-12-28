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

#include "paddle/ap/include/paddle/pir/packed_ir_op_inner_source_pattern_helper.h"
#include "paddle/ap/include/drr/drr_graph_descriptor.h"
#include "paddle/ap/include/drr/drr_node_descriptor.h"
#include "paddle/ap/include/drr/node.h"
#include "paddle/ap/include/drr/topo_kind.h"
#include "paddle/ap/include/drr/value_method_class.h"
#include "paddle/ap/include/graph/node.h"
#include "paddle/ap/include/ir_match/graph_matcher.h"
#include "paddle/ap/include/ir_match/ir_match_ctx.h"
#include "paddle/ap/include/paddle/pir_graph_descriptor.h"
#include "paddle/ap/include/paddle/pir_node.h"
#include "paddle/ap/include/paddle/pir_node_descriptor.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

namespace ap::paddle {

namespace {

std::optional<const pir::Block*> GetPirNodeBlock(const PirNode& pir_node) {
  using RetT = std::optional<const pir::Block*>;
  return pir_node.Match(
      [&](const NativeIrValue& impl) -> RetT { return std::nullopt; },
      [&](const PackedIrValue& impl) -> RetT { return std::nullopt; },
      [&](const NativeIrOpOperand& impl) -> RetT { return std::nullopt; },
      [&](const PackedIrOpOperand& impl) -> RetT { return std::nullopt; },
      [&](const RefIrOpOperand& impl) -> RetT { return std::nullopt; },
      [&](const NativeIrOp& impl) -> RetT { return impl.op->GetParent(); },
      [&](const PackedIrOp& impl) -> RetT {
        return impl.fusion_op->GetParent();
      },
      [&](const NativeIrOpResult& impl) -> RetT { return std::nullopt; },
      [&](const PackedIrOpResult& impl) -> RetT { return std::nullopt; },
      [&](const RefIrValue& impl) -> RetT { return std::nullopt; },
      [&](const RefIrOp& impl) -> RetT { return std::nullopt; },
      [&](const RefIrOpResult& impl) -> RetT { return std::nullopt; });
}

adt::Result<drr::Node> GetDrrYieldNode(
    const drr::SourcePatternCtx& src_ptn_ctx) {
  std::optional<drr::Node> yield_node;
  for (const auto& drr_node : src_ptn_ctx->node_arena->nodes()) {
    if (!drr_node.template Has<drr::NativeIrOp<drr::Node>>()) {
      continue;
    }
    ADT_LET_CONST_REF(drr_op,
                      drr_node.template TryGet<drr::NativeIrOp<drr::Node>>());
    if (drr_op->op_declare->op_name == pir::YieldOp::name()) {
      ADT_CHECK(!yield_node.has_value());
      yield_node = drr_node;
    }
  }
  ADT_CHECK(yield_node.has_value());
  return yield_node.value();
}

adt::Result<PirNode> GetPirYieldNode(const PackedIrOp& ir_op) {
  auto* block = ir_op.fusion_op.block();
  for (auto& op : *block) {
    if (op.template isa<pir::YieldOp>()) {
      return NativeIrOp{&op};
    }
  }
  return adt::errors::ValueError{"no yield op found in fusion_op block"};
}

}  // namespace

adt::Result<std::optional<ir_match::GraphMatchCtx<PirNode>>>
PackedIrOpInnerSourcePatternHelper::Match(
    const PackedIrOp& ir_op, const drr::SourcePatternCtx& src_ptn_ctx) {
  auto BelongToThisBlockOrNotOp =
      [&](const PirNode& node) -> adt::Result<bool> {
    const auto& opt_block = GetPirNodeBlock(node);
    if (!opt_block.has_value()) {
      return true;
    }
    return opt_block.value() == ir_op.fusion_op->GetParent();
  };
  using Default = drr::topo_kind::Default;
  using BlockBound = drr::topo_kind::BlockBound;
  using DrrGraphNode = graph::Node<drr::Node>;
  ap::graph::GraphDescriptor<PirNode, BlockBound> pir_graph(
      BelongToThisBlockOrNotOp);
  ap::graph::GraphDescriptor<DrrGraphNode, Default> src_ptn_graph{};
  ap::ir_match::GraphMatcher<PirNode, BlockBound, Default> graph_matcher(
      pir_graph, src_ptn_graph);
  ADT_LET_CONST_REF(drr_yield_node, GetDrrYieldNode(src_ptn_ctx));
  ADT_LET_CONST_REF(pir_yield_node, GetPirYieldNode(ir_op));
  ADT_LET_CONST_REF(
      graph_ctx,
      graph_matcher.MatchByAnchor(pir_yield_node, drr_yield_node.node()));
  ADT_LET_CONST_REF(
      graph_matched,
      graph_matcher.IsGraphMatched(graph_ctx, drr_yield_node.node()));
  if (!graph_matched) {
    return std::nullopt;
  }
  return graph_ctx;
}

}  // namespace ap::paddle
