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
#include "ap/drr/topo_kind.h"
#include "ap/graph/graph_descriptor.h"
#include "ap/graph/node.h"

namespace ap::drr {

template <typename ValueT>
struct DefaultDrrGraphDescriptor {
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

  adt::Result<bool> IsValueNode(const NodeT& node) const {
    ADT_LET_CONST_REF(drr_node, node.Get());
    return drr_node.Match(
        [&](const NativeIrValue<DrrNodeT>&) -> bool { return true; },
        [&](const PackedIrValue<DrrNodeT>&) -> bool { return true; },
        [&](const auto&) -> bool { return false; });
  }

  adt::Result<bool> Satisfy(const NodeT& node,
                            const graph::NodeCstr& node_cstr) const {
    ADT_LET_CONST_REF(drr_node, node.Get());
    const graph::NodeCstr& drr_node_cstr = drr_node.node_cstr();
    return drr_node_cstr == node_cstr;
  }
};

template <typename ValueT>
struct AllOperandAndResultDrrGraphDescriptor {
  using DrrNodeT = drr::Node<ValueT>;
  using NodeT = graph::Node<DrrNodeT>;

  DefaultDrrGraphDescriptor<ValueT> backend_graph;

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitUpstreamNodes(const NodeT& node,
                                          const DoEachT& DoEach) const {
    auto DoEachOpOrValue = [&](const NodeT& upstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(is_op_node, backend_graph.IsOpNode(upstream));
      ADT_LET_CONST_REF(is_value_node, backend_graph.IsValueNode(upstream));
      ADT_CHECK(is_op_node || is_value_node);
      return backend_graph.VisitUpstreamNodes(upstream, DoEach);
    };
    return backend_graph.VisitUpstreamNodes(node, DoEachOpOrValue);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitDownstreamNodes(const NodeT& node,
                                            const DoEachT& DoEach) const {
    auto DoEachOpOrValue =
        [&](const NodeT& downstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(is_op_node, backend_graph.IsOpNode(downstream));
      ADT_LET_CONST_REF(is_value_node, backend_graph.IsValueNode(downstream));
      ADT_CHECK(is_op_node || is_value_node);
      return backend_graph.VisitDownstreamNodes(downstream, DoEach);
    };
    return backend_graph.VisitDownstreamNodes(node, DoEachOpOrValue);
  }

  adt::Result<graph::NodeCstr> GetNodeConstraint(const NodeT& node) const {
    return backend_graph.GetNodeConstraint(node);
  }

  adt::Result<bool> IgnoredNode(const NodeT& node) const {
    ADT_LET_CONST_REF(is_op_node, backend_graph.IsOpNode(node));
    ADT_LET_CONST_REF(is_value_node, backend_graph.IsValueNode(node));
    if (is_op_node || is_value_node) {
      return true;
    }
    return backend_graph.IgnoredNode(node);
  }

  adt::Result<bool> IsOpNode(const NodeT& node) const {
    return backend_graph.IsOpNode(node);
  }

  adt::Result<bool> Satisfy(const NodeT& node,
                            const graph::NodeCstr& node_cstr) const {
    return backend_graph.Satisfy(node, node_cstr);
  }
};

template <typename ValueT>
struct NativeOperandAndResultDrrGraphDescriptor {
  using DrrNodeT = drr::Node<ValueT>;
  using NodeT = graph::Node<DrrNodeT>;

  AllOperandAndResultDrrGraphDescriptor<ValueT> backend_graph;

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitUpstreamNodes(const NodeT& node,
                                          const DoEachT& DoEach) const {
    ADT_LET_CONST_REF(is_node_native, IsNative(node));
    ADT_CHECK(is_node_native);
    auto VisitEachNative = [&](const NodeT& upstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(is_upstream_native, IsNative(upstream));
      ADT_CHECK(!is_upstream_native);
      return backend_graph.VisitUpstreamNodes(upstream, DoEach);
    };
    auto VisitEachPacked = [&](const NodeT& upstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(is_upstream_native, IsNative(upstream));
      ADT_CHECK(!is_upstream_native);
      return backend_graph.VisitUpstreamNodes(upstream, VisitEachNative);
    };
    auto DoEachOperandOrResult =
        [&](const NodeT& upstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(is_native, IsNative(upstream));
      if (is_native) {
        return DoEach(upstream);
      } else {
        return backend_graph.VisitUpstreamNodes(upstream, VisitEachPacked);
      }
    };
    return backend_graph.VisitUpstreamNodes(node, DoEachOperandOrResult);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitDownstreamNodes(const NodeT& node,
                                            const DoEachT& DoEach) const {
    ADT_LET_CONST_REF(is_node_native, IsNative(node));
    ADT_CHECK(is_node_native);
    auto VisitEachNative =
        [&](const NodeT& downstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(is_downstream_native, IsNative(downstream));
      ADT_CHECK(!is_downstream_native);
      return backend_graph.VisitDownstreamNodes(downstream, DoEach);
    };
    auto VisitEachPacked =
        [&](const NodeT& downstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(is_downstream_native, IsNative(downstream));
      ADT_CHECK(!is_downstream_native);
      return backend_graph.VisitDownstreamNodes(downstream, VisitEachNative);
    };
    auto DoEachOperandOrResult =
        [&](const NodeT& downstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(is_native, IsNative(downstream));
      if (is_native) {
        return DoEach(downstream);
      } else {
        return backend_graph.VisitDownstreamNodes(downstream, VisitEachPacked);
      }
    };
    return backend_graph.VisitDownstreamNodes(node, DoEachOperandOrResult);
  }

  adt::Result<graph::NodeCstr> GetNodeConstraint(const NodeT& node) const {
    return backend_graph.GetNodeConstraint(node);
  }

  adt::Result<bool> IgnoredNode(const NodeT& node) const {
    ADT_LET_CONST_REF(is_native, IsNative(node));
    if (!is_native) {
      return true;
    }
    return backend_graph.IgnoredNode(node);
  }

  adt::Result<bool> IsOpNode(const NodeT& node) const {
    return backend_graph.IsOpNode(node);
  }

  adt::Result<bool> Satisfy(const NodeT& node,
                            const graph::NodeCstr& node_cstr) const {
    return backend_graph.Satisfy(node, node_cstr);
  }

  adt::Result<bool> IsNative(const NodeT& node) const {
    ADT_LET_CONST_REF(drr_node, node.Get());
    return drr_node.Match(
        [&](const NativeIrOpOperand<DrrNodeT>&) -> bool { return true; },
        [&](const NativeIrOpResult<DrrNodeT>&) -> bool { return true; },
        [&](const auto&) -> bool { return false; });
  }
};

}  // namespace ap::drr

namespace ap::graph {

template <typename ValueT>
struct GraphDescriptor<graph::Node<drr::Node<ValueT>>, drr::topo_kind::Default>
    : public drr::DefaultDrrGraphDescriptor<ValueT> {};

template <typename ValueT>
struct GraphDescriptor<graph::Node<drr::Node<ValueT>>,
                       drr::topo_kind::AllOperandAndResult>
    : public drr::AllOperandAndResultDrrGraphDescriptor<ValueT> {};

template <typename ValueT>
struct GraphDescriptor<graph::Node<drr::Node<ValueT>>,
                       drr::topo_kind::NativeOperandAndResult>
    : public drr::NativeOperandAndResultDrrGraphDescriptor<ValueT> {};

}  // namespace ap::graph
