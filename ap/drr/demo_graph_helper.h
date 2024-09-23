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
#include "ap/graph/graph_matcher.h"
#include "ap/graph/node_arena.h"

namespace ap::drr {

template <typename ValueT>
struct DemoGraphHelper {
  using DrrNodeT = drr::Node<ValueT>;
  using NodeT = graph::Node<DrrNodeT>;
  using GraphDescriptor = graph::GraphDescriptor<NodeT>;

  adt::Result<bool> IsGraphMatched(
      const graph::NodeArena<DrrNodeT>& obj_node_area,
      const graph::NodeArena<DrrNodeT>& ptn_node_area) {
    GraphDescriptor big_graph{
        DrrGraphDescriptor<ValueT>{obj_node_area.shared_from_this()}};
    GraphDescriptor small_graph{
        DrrGraphDescriptor<ValueT>{ptn_node_area.shared_from_this()}};
    graph::GraphMatcher<NodeT, NodeT> graph_matcher(big_graph, small_graph);
    const auto& opt_graph_ctx = graph_matcher.Match();
    if (opt_graph_ctx.HasError()) {
      ADT_CHECK(
          opt_graph_ctx.GetError().template Has<adt::errors::MismatchError>())
          << opt_graph_ctx.GetError();
      return false;
    }
    return true;
  }
};

}  // namespace ap::drr
