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

#include "ap/drr/drr_graph_descriptor.h"
#include "ap/drr/drr_value.h"
#include "ap/drr/node.h"
#include "ap/drr/topo_kind.h"
#include "ap/graph/graph_descriptor.h"
#include "ap/graph/node.h"
#include "ap/ir_match/topo_match_ctx.h"

namespace ap::ir_match {

template <typename bg_node_t /*big graph node type*/>
struct GraphMatchCtxImpl {
  using DrrNode = drr::Node<drr::Value>;
  using sg_node_t = graph::Node<DrrNode>;

  TopoMatchCtx<bg_node_t, sg_node_t> topo_match_ctx;

  bool operator==(const GraphMatchCtxImpl& other) const {
    return this == &other;
  }
  std::size_t num_matched_bg_nodes() const {
    return topo_match_ctx->num_matched_bg_nodes();
  }

  adt::Result<bool> HasBigGraphNode(const sg_node_t& node) const {
    return topo_match_ctx->HasBigGraphNode(node);
  }

  adt::Result<bg_node_t> GetSoleBigGraphNode(const sg_node_t& node) const {
    return topo_match_ctx->GetSoleBigGraphNode(node);
  }

  adt::Result<sg_node_t> GetMatchedSmallGraphNode(
      const bg_node_t& bg_node) const {
    const auto& sg_node = topo_match_ctx->GetMatchedSmallGraphNode(bg_node);
    ADT_CHECK(sg_node.has_value());
    return sg_node.value();
  }
};

template <typename bg_node_t /*big graph node type*/>
DEFINE_ADT_RC(GraphMatchCtx, GraphMatchCtxImpl<bg_node_t>);

}  // namespace ap::ir_match
