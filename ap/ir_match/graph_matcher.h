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

template <typename bg_node_t>
struct GraphMatcher {
  using DrrNode = drr::Node<drr::Value>;
  using DrrNativeIrOp = drr::NativeIrOp<drr::Value, DrrNode>;
  using sg_node_t = graph::Node<DrrNode>;
  using DefaultTopoKind = drr::topo_kind::Default;

  TopoMatcher<bg_node_t, sg_node_t, DefaultTopoKind> topo_matcher_;

  GraphMatcher(const GraphDescriptor<bg_node_t, DefaultTopoKind>& bg_descriptor,
               const GraphDescriptor<sg_node_t, DefaultTopoKind>& sg_descriptor)
      : topo_matcher_(bg_descriptor, sg_descriptor) {}

  GraphMatcher(const GraphMatcher&) = delete;
  GraphMatcher(GraphMatcher&&) = delete;

  adt::Result<GraphMatchCtx<bg_node_t>> MatchByDefaultAnchor(
      const bg_node_t& bg_node, const sg_node_t& anchor_node) {
    ADT_LET_CONST_REF(topo_match_ctx,
                      topo_matcher_.MatchByAnchor(bg_node, anchor_node));
    return GraphMatchCtx<bg_node_t>{topo_match_ctx};
  }

  adt::Result<bool> IsGraphMatched(const GraphMatchCtx<bg_node_t>& ctx,
                                   const sg_node_t& anchor_node) const {
    return topo_matcher_.IsGraphMatched(ctx->topo_match_ctx, anchor_node);
  }
};

}  // namespace ap::ir_match
