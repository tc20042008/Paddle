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
#include "ap/drr/drr_graph_descriptor.h"
#include "ap/drr/drr_node_descriptor.h"
#include "ap/drr/node.h"
#include "ap/graph/graph_helper.h"
#include "ap/graph/node_arena.h"
#include "ap/ir_match/graph_matcher.h"

namespace ap::drr {

template <typename ValueT>
struct DemoGraphHelper {
  using DrrNodeT = drr::Node<ValueT>;
  using NodeT = graph::Node<DrrNodeT>;
  using GraphDescriptor = graph::GraphDescriptor<NodeT>;

  adt::Result<bool> IsGraphMatched(
      const graph::NodeArena<DrrNodeT>& obj_node_area,
      const graph::NodeArena<DrrNodeT>& ptn_node_area) {
    GraphDescriptor big_graph{};
    GraphDescriptor small_graph{};
    ir_match::GraphMatcher<NodeT, NodeT> graph_matcher(big_graph, small_graph);
    ADT_CHECK(ptn_node_area.nodes().size() > 0);
    const auto& start_ptn_node = ptn_node_area.nodes().at(0).node();
    ADT_LET_CONST_REF(
        anchor,
        graph::GraphHelper<NodeT>(small_graph).FindAnchor(start_ptn_node));
    ADT_LET_CONST_REF(anchor_cstr, small_graph.GetNodeConstraint(anchor));
    VLOG(10) << "anchor.node_id: "
             << graph::NodeDescriptor<NodeT>{}.DebugId(anchor);
    for (const auto& node : obj_node_area.nodes()) {
      const auto& obj_node = node.node();
      ADT_LET_CONST_REF(matched, big_graph.Satisfy(obj_node, anchor_cstr));
      if (!matched) {
        continue;
      }
      const auto& opt_graph_ctx = graph_matcher.MatchByAnchor(obj_node, anchor);
      if (opt_graph_ctx.HasOkValue()) {
        return true;
      }
      ADT_CHECK(
          opt_graph_ctx.GetError().template Has<adt::errors::MismatchError>())
          << opt_graph_ctx.GetError();
      VLOG(10) << "call stack\n"
               << opt_graph_ctx.GetError().CallStackToString() << "\n"
               << opt_graph_ctx.GetError().class_name() << ": "
               << opt_graph_ctx.GetError().msg() << " obj_node_id: "
               << graph::NodeDescriptor<NodeT>{}.DebugId(anchor);
    }
    return false;
  }
};

}  // namespace ap::drr
