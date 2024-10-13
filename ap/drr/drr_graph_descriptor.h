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
#include "ap/drr/node.h"
#include "ap/graph/graph_descriptor.h"
#include "ap/graph/node.h"

namespace ap::drr {

template <typename ValueT>
struct DrrGraphDescriptor {
  using DrrNodeT = drr::Node<ValueT>;
  using NodeT = graph::Node<DrrNodeT>;

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitUpstreamNodes(const NodeT& node,
                                          const DoEachT& DoEach) const {
    ADT_LET_CONST_REF(upstreams, node.UpstreamNodes());
    return upstreams.VisitNodes(DoEach);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitDownstreamNodes(const NodeT& node,
                                            const DoEachT& DoEach) const {
    ADT_LET_CONST_REF(downstreams, node.DownstreamNodes());
    return downstreams.VisitNodes(DoEach);
  }

  adt::Result<graph::NodeCstr> GetNodeConstraint(const NodeT& node) const {
    ADT_LET_CONST_REF(drr_node, node.Get());
    return drr_node.node_cstr();
  }

  adt::Result<bool> IgnoredNode(const NodeT& node) const {
    ADT_LET_CONST_REF(drr_node, node.Get());
    return drr_node.Match(
        [](const PackedIrValue<DrrNodeT>&) -> adt::Result<bool> {
          return true;
        },
        [&](const PackedIrOpOperand<DrrNodeT>& impl) -> adt::Result<bool> {
          ADT_LET_CONST_REF(upstreams, impl->node.UpstreamNodes());
          ADT_CHECK(upstreams.size(), 1);
          ADT_LET_CONST_REF(upstream_node, upstreams.Sole());
          return IgnoredNode(upstream_node);
        },
        [&](const PackedIrOpResult<DrrNodeT>& impl) -> adt::Result<bool> {
          ADT_LET_CONST_REF(downstreams, impl->node.DownstreamNodes());
          ADT_CHECK(downstreams.size(), 1);
          ADT_LET_CONST_REF(downstream_node, downstreams.Sole());
          return IgnoredNode(downstream_node);
        },
        [](const auto&) -> adt::Result<bool> { return false; });
  }

  adt::Result<bool> IsOpNode(const NodeT& node) const {
    ADT_LET_CONST_REF(drr_node, node.Get());
    return drr_node.Match(
        [&](const NativeIrOp<ValueT, DrrNodeT>&) -> bool { return true; },
        [&](const PackedIrOp<ValueT, DrrNodeT>&) -> bool { return true; },
        [&](const auto&) -> bool { return false; });
  }

  adt::Result<bool> Satisfy(const NodeT& node,
                            const graph::NodeCstr& node_cstr) const {
    ADT_LET_CONST_REF(drr_node, node.Get());
    const graph::NodeCstr& drr_node_cstr = drr_node.node_cstr();
    return drr_node_cstr == node_cstr;
  }
};

}  // namespace ap::drr

namespace ap::graph {

template <typename ValueT>
struct GraphDescriptor<graph::Node<drr::Node<ValueT>>>
    : public drr::DrrGraphDescriptor<ValueT> {};

}  // namespace ap::graph
