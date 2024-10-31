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

#include <map>
#include "ap/drr/drr_value.h"
#include "ap/drr/node.h"
#include "ap/drr/topo_kind.h"
#include "ap/graph/adt.h"
#include "ap/graph/node.h"
#include "ap/ir_match/graph_match_ctx.h"
#include "ap/ir_match/topo_matcher.h"

namespace ap::ir_match {

template <typename bg_node_t, typename BGTopoKind, typename SGTopoKind>
struct GraphMatcher {
  using DrrNode = drr::Node<drr::Value>;
  using DrrNativeIrOp = drr::NativeIrOp<drr::Value, DrrNode>;
  using sg_node_t = graph::Node<DrrNode>;

  TopoMatcher<bg_node_t, sg_node_t, BGTopoKind, SGTopoKind> topo_matcher_;

  GraphMatcher(const GraphDescriptor<bg_node_t, BGTopoKind>& bg_descriptor,
               const GraphDescriptor<sg_node_t, SGTopoKind>& sg_descriptor)
      : topo_matcher_(bg_descriptor, sg_descriptor) {}

  GraphMatcher(const GraphMatcher&) = delete;
  GraphMatcher(GraphMatcher&&) = delete;

  adt::Result<GraphMatchCtx<bg_node_t>> MatchByAnchor(
      const bg_node_t& bg_node, const sg_node_t& anchor_node) {
    ADT_LET_CONST_REF(topo_match_ctx,
                      topo_matcher_.MatchByAnchor(bg_node, anchor_node));
    return GraphMatchCtx<bg_node_t>{topo_match_ctx};
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitMisMatchedNodes(
      const GraphMatchCtx<bg_node_t>& graph_match_ctx,
      const sg_node_t& anchor_node,
      const DoEachT& DoEach) const {
    const auto& topo_match_ctx = graph_match_ctx->topo_match_ctx;
    return topo_matcher_.VisitMisMatchedNodes(
        topo_match_ctx, anchor_node, DoEach);
  }

  adt::Result<adt::Ok> UpdateByConnectionsUntilDone(
      GraphMatchCtx<bg_node_t>* ctx, const sg_node_t& anchor_node) {
    return topo_matcher_.UpdateByConnectionsUntilDone(&*(*ctx)->topo_match_ctx,
                                                      anchor_node);
  }

  adt::Result<bool> IsGraphMatched(const GraphMatchCtx<bg_node_t>& ctx,
                                   const sg_node_t& anchor_node) const {
    return topo_matcher_.IsGraphMatched(ctx->topo_match_ctx, anchor_node);
  }
};

}  // namespace ap::ir_match
