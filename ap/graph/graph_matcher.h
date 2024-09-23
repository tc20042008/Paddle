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
#include "ap/graph/graph_match_ctx.h"
#include "ap/graph/node.h"
#include "ap/graph/node_arena.h"
#include "ap/graph/topo_path_ptn_hashs.h"
#include "glog/logging.h"

namespace ap::graph {

template <typename bg_node_t, typename sg_node_t>
struct GraphMatcher {
  GraphMatcher(const GraphDescriptor<bg_node_t>& bg_descriptor,
               const GraphDescriptor<sg_node_t>& sg_descriptor)
      : bg_descriptor_(bg_descriptor), sg_descriptor_(sg_descriptor) {}

  GraphMatcher(const GraphMatcher&) = delete;
  GraphMatcher(GraphMatcher&&) = delete;

  adt::Result<GraphMatchCtx<bg_node_t, sg_node_t>> Match() {
    ADT_LET_CONST_REF(anchor, FindAnchor());
    VLOG(10) << "anchor.node_id: "
             << NodeDescriptor<sg_node_t>{}.DebugId(anchor);
    ADT_LET_CONST_REF(anchor_cstr, sg_descriptor_.GetNodeConstraint(anchor));
    std::optional<GraphMatchCtx<bg_node_t, sg_node_t>> ret;
    const auto& TryMatchBgNode =
        [&](const bg_node_t& bg_node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(matched, bg_descriptor_.Satisfy(bg_node, anchor_cstr));
      if (!matched) {
        return adt::Ok{};
      }
      const auto& opt_graph_ctx = MatchByAnchor(bg_node, anchor);
      if (opt_graph_ctx.HasError()) {
        ADT_CHECK(
            opt_graph_ctx.GetError().template Has<adt::errors::MismatchError>())
            << opt_graph_ctx.GetError();
        VLOG(10) << opt_graph_ctx.GetError().class_name() << ": "
                 << opt_graph_ctx.GetError().msg() << " bg_node_id: "
                 << NodeDescriptor<bg_node_t>{}.DebugId(anchor);
        return adt::Ok{};
      }
      ret = opt_graph_ctx.GetOkValue();
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(bg_descriptor_.VisitAllNodes(TryMatchBgNode));
    ADT_CHECK(ret.has_value())
        << adt::errors::MismatchError{"no anchor matched."};
    return ret.value();
  }

  adt::Result<sg_node_t> FindAnchor() {
    const auto topo_walker = GetSmallGraphTopoWalker();
    const auto IsSource = [&](const sg_node_t& sg_node) -> adt::Result<bool> {
      bool has_source = false;
      auto SetHasSource = [&](const sg_node_t&) -> adt::Result<adt::Ok> {
        has_source = true;
        return adt::Ok{};
      };
      ADT_RETURN_IF_ERR(
          sg_descriptor_.VisitUpstreamNodes(sg_node, SetHasSource));
      return !has_source;
    };
    const auto IsSink = [&](const sg_node_t& sg_node) -> adt::Result<bool> {
      bool has_sink = false;
      auto SetHasSink = [&](const sg_node_t&) -> adt::Result<adt::Ok> {
        has_sink = true;
        return adt::Ok{};
      };
      ADT_RETURN_IF_ERR(
          sg_descriptor_.VisitDownstreamNodes(sg_node, SetHasSink));
      return !has_sink;
    };
    std::unordered_set<sg_node_t> starts;
    auto CollectStarts = [&](const sg_node_t& sg_node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(ignored, sg_descriptor_.IgnoredNode(sg_node));
      if (ignored) {
        return adt::Ok{};
      }
      ADT_LET_CONST_REF(is_source, IsSource(sg_node));
      ADT_LET_CONST_REF(is_sink, IsSink(sg_node));
      if (is_source || is_sink) {
        starts.insert(sg_node);
      }
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(sg_descriptor_.VisitAllNodes(CollectStarts));
    ADT_CHECK(starts.size() > 0);
    const auto bfs_walker = GetSmallGraphBfsWalker();
    std::unordered_map<sg_node_t, size_t> node2depth;
    std::map<size_t, std::vector<sg_node_t>> depth2nodes;
    auto UpdateNodeDepth =
        [&](const sg_node_t& sg_node) -> adt::Result<adt::Ok> {
      size_t max_depth = 0;
      ADT_RETURN_IF_ERR(bfs_walker.VisitNextNodes(
          sg_node, [&](const sg_node_t& prev) -> adt::Result<adt::Ok> {
            const auto& iter = node2depth.find(prev);
            if (iter != node2depth.end()) {
              max_depth = std::max(max_depth, iter->second);
            }
            return adt::Ok{};
          }));
      node2depth[sg_node] = max_depth;
      depth2nodes[max_depth].push_back(sg_node);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(
        bfs_walker(starts.begin(), starts.end(), UpdateNodeDepth));
    const auto& last = depth2nodes.rbegin();
    ADT_CHECK(last != depth2nodes.rend());
    ADT_CHECK(last->second.size() > 0);
    return last->second.at(0);
  }

  adt::Result<GraphMatchCtx<bg_node_t, sg_node_t>> MatchByAnchor(
      const bg_node_t& bg_node, const sg_node_t& anchor_node) {
    ADT_LET_CONST_REF(graph_match_ctx,
                      MakeGraphMatchCtxFromAnchor(bg_node, anchor_node));
    ADT_RETURN_IF_ERR(UpdateByConnectionsUntilDone(
        &*graph_match_ctx.shared_ptr(), anchor_node));
    return graph_match_ctx;
  }

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

  adt::Result<bool> IsGraphMatched(
      const GraphMatchCtx<bg_node_t, sg_node_t>& ctx,
      const sg_node_t& anchor_node) {
    adt::BfsWalker<sg_node_t> bfs_walker = GetSmallGraphBfsWalker();
    std::size_t num_sg_nodes = 0;
    auto AccNumSgNodes = [&](const sg_node_t& sg_node) -> adt::Result<adt::Ok> {
      ADT_CHECK(ctx->HasObjNode(sg_node))
          << adt::errors::MismatchError{"IsGraphMatched: sg_node not matched."};
      ADT_LET_CONST_REF(bg_nodes, ctx->GetObjNodes(sg_node));
      ADT_CHECK(bg_nodes.size() == 1) << adt::errors::MismatchError{
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

  adt::BfsWalker<sg_node_t> GetSmallGraphBfsWalker() {
    auto small_graph = this->sg_descriptor_;
    const auto& ForEachNext =
        [small_graph](const sg_node_t& node,
                      const auto& VisitNext) -> adt::Result<adt::Ok> {
      auto DoEach = [&](const sg_node_t& next) -> adt::Result<adt::Ok> {
        ADT_LET_CONST_REF(is_ignored, small_graph.IgnoredNode(next));
        if (is_ignored) {
          return adt::Ok{};
        }
        return VisitNext(next);
      };
      ADT_RETURN_IF_ERR(small_graph.VisitDownstreamNodes(node, DoEach));
      ADT_RETURN_IF_ERR(small_graph.VisitUpstreamNodes(node, DoEach));
      return adt::Ok{};
    };
    return adt::BfsWalker<sg_node_t>(ForEachNext);
  }

  adt::TopoWalker<sg_node_t> GetSmallGraphTopoWalker() {
    auto small_graph = this->sg_descriptor_;
    const auto& ForEachPrev =
        [small_graph](const sg_node_t& node,
                      const auto& VisitPrev) -> adt::Result<adt::Ok> {
      auto DoEach = [&](const sg_node_t& prev) -> adt::Result<adt::Ok> {
        ADT_LET_CONST_REF(is_ignored, small_graph.IgnoredNode(prev));
        if (is_ignored) {
          return adt::Ok{};
        }
        return VisitPrev(prev);
      };
      return small_graph.VisitUpstreamNodes(node, DoEach);
    };
    const auto& ForEachNext =
        [small_graph](const sg_node_t& node,
                      const auto& VisitNext) -> adt::Result<adt::Ok> {
      auto DoEach = [&](const sg_node_t& next) -> adt::Result<adt::Ok> {
        ADT_LET_CONST_REF(is_ignored, small_graph.IgnoredNode(next));
        if (is_ignored) {
          return adt::Ok{};
        }
        return VisitNext(next);
      };
      return small_graph.VisitDownstreamNodes(node, DoEach);
    };
    return adt::TopoWalker<sg_node_t>(ForEachPrev, ForEachNext);
  }

 private:
  adt::Result<GraphMatchCtx<bg_node_t, sg_node_t>> MakeGraphMatchCtxFromAnchor(
      const bg_node_t& bg_node, const sg_node_t& anchor_node) {
    GraphMatchCtx<bg_node_t, sg_node_t> match_ctx{};
    const auto& ptn_bfs_walker = GetSmallGraphBfsWalker();
    auto InitMatchCtx = [&](const sg_node_t& sg_node) -> adt::Result<adt::Ok> {
      if (sg_node == anchor_node) {
        SmallSet<bg_node_t> bg_nodes;
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
    const auto& ptn_bfs_walker = GetSmallGraphBfsWalker();
    bool updated = false;
    auto Update = [&](const sg_node_t& sg_node) -> adt::Result<adt::Ok> {
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
    auto Update = [&](const sg_node_t& node,
                      tIsUpstream<bool> is_upstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(
          bg_nodes,
          GetMatchedObjNodesFromConnected(*ctx, sg_node, node, is_upstream));
      ADT_RETURN_IF_ERR(ctx->UpdateObjNodes(sg_node, bg_nodes));
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(ForEachInitedUpstream(*ctx, sg_node, Update));
    ADT_RETURN_IF_ERR(ForEachInitedDownstream(*ctx, sg_node, Update));
    return old_num_bg_nodes != bg_nodes_ptr->size();
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

  adt::Result<SmallSet<bg_node_t>> GetMatchedObjNodesFromConnected(
      const GraphMatchCtxImpl<bg_node_t, sg_node_t>& ctx,
      const sg_node_t& sg_node,
      const sg_node_t& from_node,
      tIsUpstream<bool> is_from_node_upstream) {
    SmallSet<bg_node_t> bg_nodes;
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

}  // namespace ap::graph
