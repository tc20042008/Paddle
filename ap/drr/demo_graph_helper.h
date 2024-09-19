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
#include "ap/drr/demo_graph_match_util.h"
#include "ap/drr/node.h"
#include "ap/graph/graph_matcher.h"
#include "ap/graph/node_arena.h"

namespace ap::drr {

template <typename ValueT>
struct DemoGraphHelper {
  using NodeT = drr::Node<ValueT>;

  adt::Result<bool> IsGraphMatched(
      const graph::NodeArena<NodeT>& obj_node_area,
      const graph::NodeArena<NodeT>& ptn_node_area) {
    graph::GraphMatcher<NodeT, NodeT> graph_matcher{
        &IsIgnoredPackedNode<ValueT>, &IsObjMatch<ValueT>};
    const auto& opt_graph_ctx =
        graph_matcher.Match(obj_node_area, ptn_node_area);
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
