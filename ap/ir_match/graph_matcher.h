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
#include "ap/graph/adt.h"
#include "ap/graph/graph_descriptor.h"
#include "ap/graph/graph_helper.h"
#include "ap/graph/node.h"
#include "ap/graph/node_arena.h"
#include "ap/graph/topo_path_ptn_hashs.h"
#include "ap/ir_match/graph_match_ctx.h"
#include "ap/ir_match/tags.h"
#include "glog/logging.h"

namespace ap::ir_match {

using graph::GraphDescriptor;
using graph::GraphHelper;

template <typename bg_node_t, typename sg_node_t>
struct GraphMatcher {
  GraphMatcher(const GraphDescriptor<bg_node_t>& bg_descriptor,
               const GraphDescriptor<sg_node_t>& sg_descriptor)
      : bg_descriptor_(bg_descriptor), sg_descriptor_(sg_descriptor) {}

  GraphMatcher(const GraphMatcher&) = delete;
  GraphMatcher(GraphMatcher&&) = delete;

  adt::Result<GraphMatchCtx<bg_node_t, sg_node_t>> MatchByAnchor(
      const bg_node_t& bg_node, const sg_node_t& anchor_node) {
    ADT_LET_CONST_REF(graph_match_ctx,
                      MakeGraphMatchCtxFromAnchor(bg_node, anchor_node));
    ADT_RETURN_IF_ERR(UpdateByConnectionsUntilDone(
        &*graph_match_ctx.shared_ptr(), anchor_node));
    return graph_match_ctx;
  }

  adt::Result<bool> IsGraphMatched(
      const GraphMatchCtx<bg_node_t, sg_node_t>& ctx,
      const sg_node_t& anchor_node) {
    adt::BfsWalker<sg_node_t> bfs_walker =
        GraphHelper<sg_node_t>(sg_descriptor_).GetBfsWalker();
    std::size_t num_sg_nodes = 0;
    auto AccNumSgNodes = [&](const sg_node_t& sg_node) -> adt::Result<adt::Ok> {
      ADT_CHECK(ctx->HasObjNode(sg_node))
          << adt::errors::MismatchError{"IsGraphMatched: sg_node not matched."};
      ADT_LET_CONST_REF(bg_nodes, ctx->GetObjNodes(sg_node));
      ADT_CHECK(bg_nodes->size() == 1) << adt::errors::MismatchError{
          "IsGraphMatched: more than 1 bg_nodes matched to one sg_node."};
      ++num_sg_nodes;
      return adt::Ok{};
    };
    const auto& ret = bfs_walker(anchor_node, AccNumSgNodes);
    if (ret.HasError()) {
      ADT_CHECK(ret.GetError().template Has<adt::errors::MismatchError>())
          << ret.GetError();
      return false;
    }
    return num_sg_nodes == ctx->num_matched_bg_nodes();
  }

 private:
  adt::Result<adt::Ok> UpdateByConnectionsUntilDone(
      GraphMatchCtxImpl<bg_node_t, sg_node_t>* ctx,
      const sg_node_t& anchor_node) {
    size_t kDeadloopDectionSize = 999999;
    while (true) {
      ADT_LET_CONST_REF(updated, UpdateAllByConnections(ctx, anchor_node));
      if (!updated) {
        break;
      }
      if (--kDeadloopDectionSize <= 0) {
        return adt::errors::RuntimeError{"Dead loop detected."};
      }
    }
    return adt::Ok{};
  }

  adt::Result<GraphMatchCtx<bg_node_t, sg_node_t>> MakeGraphMatchCtxFromAnchor(
      const bg_node_t& bg_node, const sg_node_t& anchor_node) {
    GraphMatchCtx<bg_node_t, sg_node_t> match_ctx{};
    const auto& ptn_bfs_walker =
        GraphHelper<sg_node_t>(sg_descriptor_).GetBfsWalker();
    auto InitMatchCtx = [&](const sg_node_t& sg_node) -> adt::Result<adt::Ok> {
      if (sg_node == anchor_node) {
        std::unordered_set<bg_node_t> bg_nodes;
        bg_nodes.insert(bg_node);
        ADT_RETURN_IF_ERR(match_ctx->InitObjNodes(anchor_node, bg_nodes));
      } else {
        ADT_RETURN_IF_ERR(GraphMatchCtxInitNode(&*match_ctx, sg_node));
      }
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(ptn_bfs_walker(anchor_node, InitMatchCtx));
    return match_ctx;
  }

  adt::Result<bool> UpdateAllByConnections(
      GraphMatchCtxImpl<bg_node_t, sg_node_t>* ctx,
      const sg_node_t& anchor_node) {
    const auto& ptn_bfs_walker =
        GraphHelper<sg_node_t>(sg_descriptor_).GetBfsWalker();
    bool updated = false;
    auto Update = [&](const sg_node_t& sg_node) -> adt::Result<adt::Ok> {
      // no need to update anchor_node.
      if (anchor_node == sg_node) {
        return adt::Ok{};
      }
      ADT_LET_CONST_REF(current_updated, UpdateByConnections(ctx, sg_node));
      updated = updated || current_updated;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(ptn_bfs_walker(anchor_node, Update));
    return updated;
  }

  adt::Result<bool> UpdateByConnections(
      GraphMatchCtxImpl<bg_node_t, sg_node_t>* ctx, const sg_node_t& sg_node) {
    ADT_LET_CONST_REF(bg_nodes_ptr, ctx->GetObjNodes(sg_node));
    const size_t old_num_bg_nodes = bg_nodes_ptr->size();
    auto Update = [&](const sg_node_t& nearby_node,
                      tIsUpstream<bool> is_upstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(bg_nodes,
                        GetMatchedObjNodesFromConnected(
                            *ctx, sg_node, nearby_node, is_upstream));
      ADT_CHECK(!bg_nodes.empty()) << adt::errors::RuntimeError{
          std::string() + "small_graph_node: " +
          graph::NodeDescriptor<sg_node_t>{}.DebugId(sg_node) +
          ", old_big_graph_nodes: " + GetNodesDebugIds(bg_nodes_ptr) +
          ", nearby_node: " +
          graph::NodeDescriptor<sg_node_t>{}.DebugId(nearby_node) +
          ", is_nearby_node_from_upstream: " +
          std::to_string(is_upstream.value())};
      ADT_RETURN_IF_ERR(ctx->UpdateObjNodes(sg_node, bg_nodes));
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(ForEachInitedUpstream(*ctx, sg_node, Update));
    ADT_RETURN_IF_ERR(ForEachInitedDownstream(*ctx, sg_node, Update));
    return old_num_bg_nodes != bg_nodes_ptr->size();
  }

  std::string GetNodesDebugIds(
      const std::unordered_set<bg_node_t>* nodes) const {
    std::ostringstream ss;
    int i = 0;
    for (const auto& node : *nodes) {
      if (i++ > 0) {
        ss << " ";
      }
      ss << graph::NodeDescriptor<bg_node_t>{}.DebugId(node);
    }
    return ss.str();
  }

  adt::Result<adt::Ok> GraphMatchCtxInitNode(
      GraphMatchCtxImpl<bg_node_t, sg_node_t>* ctx, const sg_node_t& sg_node) {
    ADT_CHECK(!ctx->HasObjNode(sg_node));
    bool inited = false;
    auto InitOrUpdate =
        [&](const sg_node_t& node,
            tIsUpstream<bool> is_upstream) -> adt::Result<adt::Ok> {
      if (!inited) {
        ADT_LET_CONST_REF(
            bg_nodes,
            GetMatchedObjNodesFromConnected(*ctx, sg_node, node, is_upstream));
        ADT_RETURN_IF_ERR(ctx->InitObjNodes(sg_node, bg_nodes));
        inited = (bg_nodes.size() > 0);
      } else {
        ADT_LET_CONST_REF(
            bg_nodes,
            GetMatchedObjNodesFromConnected(*ctx, sg_node, node, is_upstream));
        ADT_RETURN_IF_ERR(ctx->UpdateObjNodes(sg_node, bg_nodes));
      }
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(ForEachInitedUpstream(*ctx, sg_node, InitOrUpdate));
    ADT_RETURN_IF_ERR(ForEachInitedDownstream(*ctx, sg_node, InitOrUpdate));
    ADT_CHECK(inited) << adt::errors::MismatchError{
        "sg_node not successfully inited."};
    return adt::Ok{};
  }

  adt::Result<std::unordered_set<bg_node_t>> GetMatchedObjNodesFromConnected(
      const GraphMatchCtxImpl<bg_node_t, sg_node_t>& ctx,
      const sg_node_t& sg_node,
      const sg_node_t& from_node,
      tIsUpstream<bool> is_from_node_upstream) {
    std::unordered_set<bg_node_t> bg_nodes;
    const auto& DoEachMatched =
        [&](const bg_node_t& bg_node) -> adt::Result<adt::Ok> {
      bg_nodes.insert(bg_node);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitMatchedObjNodesFromConnected(
        ctx, sg_node, from_node, is_from_node_upstream, DoEachMatched));
    return bg_nodes;
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitMatchedObjNodesFromConnected(
      const GraphMatchCtxImpl<bg_node_t, sg_node_t>& ctx,
      const sg_node_t& sg_node,
      const sg_node_t& from_node,
      tIsUpstream<bool> is_from_node_upstream,
      const DoEachT& DoEach) {
    ADT_LET_CONST_REF(sg_node_cstr, sg_descriptor_.GetNodeConstraint(sg_node));
    const auto& VisitObjNode =
        [&](const bg_node_t& bg_node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(matched, bg_descriptor_.Satisfy(bg_node, sg_node_cstr));
      if (!matched) {
        return adt::Ok{};
      }
      const auto& opt_matched_sg_node = ctx.GetMatchedPtnNode(bg_node);
      if (!opt_matched_sg_node.has_value() ||
          opt_matched_sg_node.value() == sg_node) {
        return DoEach(bg_node);
      }
      return adt::Ok{};
    };
    ADT_LET_CONST_REF(from_bg_nodes_ptr, ctx.GetObjNodes(from_node));
    for (const bg_node_t& from_bg_node : *from_bg_nodes_ptr) {
      if (is_from_node_upstream.value()) {
        ADT_RETURN_IF_ERR(
            bg_descriptor_.VisitDownstreamNodes(from_bg_node, VisitObjNode));
      } else {
        ADT_RETURN_IF_ERR(
            bg_descriptor_.VisitUpstreamNodes(from_bg_node, VisitObjNode));
      }
    }
    return adt::Ok{};
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> ForEachInitedUpstream(
      const GraphMatchCtxImpl<bg_node_t, sg_node_t>& ctx,
      const sg_node_t& sg_node,
      const DoEachT& DoEach) {
    auto Visit = [&](const sg_node_t& src) -> adt::Result<adt::Ok> {
      if (ctx.HasObjNode(src)) {
        return DoEach(src, tIsUpstream<bool>{true});
      }
      return adt::Ok{};
    };
    return sg_descriptor_.VisitUpstreamNodes(sg_node, Visit);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> ForEachInitedDownstream(
      const GraphMatchCtxImpl<bg_node_t, sg_node_t>& ctx,
      const sg_node_t& sg_node,
      const DoEachT& DoEach) {
    auto Visit = [&](const sg_node_t& dst) -> adt::Result<adt::Ok> {
      if (ctx.HasObjNode(dst)) {
        return DoEach(dst, tIsUpstream<bool>{false});
      }
      return adt::Ok{};
    };
    return sg_descriptor_.VisitDownstreamNodes(sg_node, Visit);
  }

  GraphDescriptor<bg_node_t> bg_descriptor_;
  GraphDescriptor<sg_node_t> sg_descriptor_;
};

}  // namespace ap::ir_match
