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
#include "ap/graph/graph_match_ctx.h"
#include "ap/graph/node.h"
#include "ap/graph/node_arena.h"
#include "ap/graph/topo_path_ptn_hashs.h"
#include "glog/logging.h"

namespace ap::graph {

template <typename ObjT, typename PtnT>
struct GraphMatcher {
  using obj_node_t = graph::Node<ObjT>;
  using ptn_node_t = graph::Node<PtnT>;

  GraphMatcher(
      const std::function<adt::Result<bool>(const PtnT&)>& IgnoredPtnVal,
      const std::function<adt::Result<bool>(const ObjT&, const PtnT&)>&
          ObjMatchVal)
      : IgnoredPtn(IgnoredPtnVal), ObjMatch(ObjMatchVal) {}

  adt::Result<GraphMatchCtx<ObjT, PtnT>> Match(
      const NodeArena<ObjT>& obj_node_arena,
      const NodeArena<PtnT>& ptn_node_arena) {
    ADT_LET_CONST_REF(anchor, FindAnchor(ptn_node_arena));
    VLOG(10) << "anchor.node_id: " << anchor.node().node_id().value();
    for (const auto& obj : obj_node_arena.nodes()) {
      ADT_LET_CONST_REF(matched, ObjMatch(obj, anchor));
      if (matched) {
        const auto& opt_graph_ctx = MatchByAnchor(obj.node(), anchor.node());
        if (opt_graph_ctx.HasOkValue()) {
          return opt_graph_ctx.GetOkValue();
        } else {
          ADT_CHECK(opt_graph_ctx.GetError()
                        .template Has<adt::errors::MismatchError>())
              << opt_graph_ctx.GetError();
          VLOG(10) << opt_graph_ctx.GetError().class_name() << ": "
                   << opt_graph_ctx.GetError().msg()
                   << " obj_node_id: " << obj.node().node_id().value();
        }
      }
    }
    return adt::errors::MismatchError{"no anchor matched."};
  }

  adt::Result<PtnT> FindAnchor(const NodeArena<PtnT>& ptn_node_arena) {
    const auto topo_walker = GetPtnTopoWalker();
    const auto IsSource = [](const PtnT& ptn) -> adt::Result<bool> {
      ADT_LET_CONST_REF(upstreams, ptn.node().UpstreamNodes());
      return upstreams.size() == 0;
    };
    const auto IsSink = [](const PtnT& ptn) -> adt::Result<bool> {
      ADT_LET_CONST_REF(downstreams, ptn.node().DownstreamNodes());
      return downstreams.size() == 0;
    };
    ADT_LET_CONST_REF(starts,
                      [&]() -> adt::Result<std::unordered_set<ptn_node_t>> {
                        std::unordered_set<ptn_node_t> starts;
                        for (const auto& ptn : ptn_node_arena.nodes()) {
                          ADT_LET_CONST_REF(ignored, IgnoredPtn(ptn));
                          if (ignored) {
                            continue;
                          }
                          ADT_LET_CONST_REF(is_source, IsSource(ptn));
                          ADT_LET_CONST_REF(is_sink, IsSink(ptn));
                          if (is_source || is_sink) {
                            starts.insert(ptn.node());
                          }
                        }
                        return starts;
                      }());
    ADT_CHECK(starts.size() > 0);
    const auto bfs_walker = GetPtnBfsWalker();
    std::unordered_map<ptn_node_t, size_t> node2depth;
    std::map<size_t, std::vector<ptn_node_t>> depth2nodes;
    ADT_RETURN_IF_ERR(bfs_walker(
        starts.begin(),
        starts.end(),
        [&](const ptn_node_t& ptn_node) -> adt::Result<adt::Ok> {
          size_t max_depth = 0;
          ADT_RETURN_IF_ERR(bfs_walker.VisitNextNodes(
              ptn_node, [&](const ptn_node_t& prev) -> adt::Result<adt::Ok> {
                const auto& iter = node2depth.find(prev);
                if (iter != node2depth.end()) {
                  max_depth = std::max(max_depth, iter->second);
                }
                return adt::Ok{};
              }));
          node2depth[ptn_node] = max_depth;
          depth2nodes[max_depth].push_back(ptn_node);
          return adt::Ok{};
        }));
    const auto& last = depth2nodes.rbegin();
    ADT_CHECK(last != depth2nodes.rend());
    ADT_CHECK(last->second.size() > 0);
    return last->second.at(0).Get();
  }

  adt::Result<GraphMatchCtx<ObjT, PtnT>> MatchByAnchor(
      const obj_node_t& obj_node, const ptn_node_t& anchor_node) {
    ADT_LET_CONST_REF(anchor_matched, ObjNodeMatch(obj_node, anchor_node));
    ADT_CHECK(anchor_matched);
    ADT_LET_CONST_REF(graph_match_ctx,
                      MakeGraphMatchCtxFromAnchor(obj_node, anchor_node));
    ADT_RETURN_IF_ERR(UpdateByConnectionsUntilDone(
        &*graph_match_ctx.shared_ptr(), anchor_node));
    return graph_match_ctx;
  }

  adt::Result<adt::Ok> UpdateByConnectionsUntilDone(
      GraphMatchCtxImpl<ObjT, PtnT>* ctx, const ptn_node_t& anchor_node) {
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

  adt::Result<bool> IsGraphMatched(const GraphMatchCtx<ObjT, PtnT>& ctx,
                                   const ptn_node_t& anchor_node) {
    adt::BfsWalker<ptn_node_t> bfs_walker = GetPtnBfsWalker();
    std::size_t num_ptn_nodes = 0;
    const auto& ret = bfs_walker(
        anchor_node, [&](const ptn_node_t& ptn_node) -> adt::Result<adt::Ok> {
          ADT_CHECK(ctx->HasObjNode(ptn_node)) << adt::errors::MismatchError{
              "IsGraphMatched: ptn_node not matched."};
          ADT_LET_CONST_REF(obj_nodes, ctx->GetObjNodes(ptn_node));
          ADT_CHECK(obj_nodes.size() == 1) << adt::errors::MismatchError{
              "IsGraphMatched: more than 1 obj_nodes matched to one ptn_node."};
          ++num_ptn_nodes;
          return adt::Ok{};
        });
    if (ret.HasError()) {
      ADT_CHECK(ret.GetError().template Has<adt::errors::MismatchError>())
          << ret.GetError();
      return false;
    }
    return num_ptn_nodes == ctx->num_matched_obj_nodes();
  }

  adt::BfsWalker<ptn_node_t> GetPtnBfsWalker() {
    const auto& IgnoredPtnFunc = this->IgnoredPtn;
    const auto& ForEachNext =
        [IgnoredPtnFunc](const ptn_node_t& node,
                         const auto& VisitNext) -> adt::Result<adt::Ok> {
      auto DoEach = [&](const ptn_node_t& next) -> adt::Result<adt::Ok> {
        ADT_LET_CONST_REF(next_obj, next.Get());
        ADT_LET_CONST_REF(is_ignored, IgnoredPtnFunc(next_obj));
        if (is_ignored) {
          return adt::Ok{};
        }
        return VisitNext(next);
      };
      ADT_LET_CONST_REF(downstream_nodes, node.DownstreamNodes());
      ADT_RETURN_IF_ERR(downstream_nodes.VisitNodes(DoEach));
      ADT_LET_CONST_REF(upstream_nodes, node.UpstreamNodes());
      ADT_RETURN_IF_ERR(upstream_nodes.VisitNodes(DoEach));
      return adt::Ok{};
    };
    return adt::BfsWalker<ptn_node_t>(ForEachNext);
  }

  adt::TopoWalker<ptn_node_t> GetPtnTopoWalker() {
    const auto& IgnoredPtnFunc = this->IgnoredPtn;
    const auto& ForEachPrev =
        [IgnoredPtnFunc](const ptn_node_t& node,
                         const auto& VisitNext) -> adt::Result<adt::Ok> {
      auto DoEach = [&](const ptn_node_t& next) -> adt::Result<adt::Ok> {
        ADT_LET_CONST_REF(next_obj, next.Get());
        ADT_LET_CONST_REF(is_ignored, IgnoredPtnFunc(next_obj));
        if (is_ignored) {
          return adt::Ok{};
        }
        return VisitNext(next);
      };
      ADT_LET_CONST_REF(upstream_nodes, node.UpstreamNodes());
      ADT_RETURN_IF_ERR(upstream_nodes.VisitNodes(DoEach));
      return adt::Ok{};
    };
    const auto& ForEachNext =
        [IgnoredPtnFunc](const ptn_node_t& node,
                         const auto& VisitNext) -> adt::Result<adt::Ok> {
      auto DoEach = [&](const ptn_node_t& next) -> adt::Result<adt::Ok> {
        ADT_LET_CONST_REF(next_obj, next.Get());
        ADT_LET_CONST_REF(is_ignored, IgnoredPtnFunc(next_obj));
        if (is_ignored) {
          return adt::Ok{};
        }
        return VisitNext(next);
      };
      ADT_LET_CONST_REF(downstream_nodes, node.DownstreamNodes());
      ADT_RETURN_IF_ERR(downstream_nodes.VisitNodes(DoEach));
      return adt::Ok{};
    };
    return adt::TopoWalker<ptn_node_t>(ForEachPrev, ForEachNext);
  }

 private:
  adt::Result<GraphMatchCtx<ObjT, PtnT>> MakeGraphMatchCtxFromAnchor(
      const obj_node_t& obj_node, const ptn_node_t& anchor_node) {
    GraphMatchCtx<ObjT, PtnT> match_ctx{};
    const auto& ptn_bfs_walker = GetPtnBfsWalker();
    const auto& walker_ret = ptn_bfs_walker(
        anchor_node, [&](const ptn_node_t& ptn_node) -> adt::Result<adt::Ok> {
          if (ptn_node == anchor_node) {
            SmallSet<obj_node_t> obj_nodes;
            obj_nodes.insert(obj_node);
            ADT_RETURN_IF_ERR(match_ctx->InitObjNodes(anchor_node, obj_nodes));
          } else {
            ADT_RETURN_IF_ERR(GraphMatchCtxInitNode(&*match_ctx, ptn_node));
          }
          return adt::Ok{};
        });
    ADT_RETURN_IF_ERR(walker_ret);
    return match_ctx;
  }

  adt::Result<bool> UpdateAllByConnections(GraphMatchCtxImpl<ObjT, PtnT>* ctx,
                                           const ptn_node_t& anchor_node) {
    const auto& ptn_bfs_walker = GetPtnBfsWalker();
    bool updated = false;
    ADT_RETURN_IF_ERR(ptn_bfs_walker(
        anchor_node, [&](const ptn_node_t& ptn_node) -> adt::Result<adt::Ok> {
          ADT_LET_CONST_REF(current_updated,
                            UpdateByConnections(ctx, ptn_node));
          updated = updated || current_updated;
          return adt::Ok{};
        }));
    return updated;
  }

  adt::Result<bool> UpdateByConnections(GraphMatchCtxImpl<ObjT, PtnT>* ctx,
                                        const ptn_node_t& ptn_node) {
    ADT_LET_CONST_REF(obj_nodes_ptr, ctx->GetObjNodes(ptn_node));
    const size_t old_num_obj_nodes = obj_nodes_ptr->size();
    auto Update = [&](const ptn_node_t& node,
                      tIsUpstream<bool> is_upstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(
          obj_nodes,
          GetMatchedObjNodesFromConnected(*ctx, ptn_node, node, is_upstream));
      ADT_RETURN_IF_ERR(ctx->UpdateObjNodes(ptn_node, obj_nodes));
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(ForEachInitedUpstream(*ctx, ptn_node, Update));
    ADT_RETURN_IF_ERR(ForEachInitedDownstream(*ctx, ptn_node, Update));
    return old_num_obj_nodes != obj_nodes_ptr->size();
  }

  adt::Result<adt::Ok> GraphMatchCtxInitNode(GraphMatchCtxImpl<ObjT, PtnT>* ctx,
                                             const ptn_node_t& ptn_node) {
    ADT_CHECK(!ctx->HasObjNode(ptn_node));
    bool inited = false;
    auto InitOrUpdate =
        [&](const ptn_node_t& node,
            tIsUpstream<bool> is_upstream) -> adt::Result<adt::Ok> {
      if (!inited) {
        ADT_LET_CONST_REF(
            obj_nodes,
            GetMatchedObjNodesFromConnected(*ctx, ptn_node, node, is_upstream));
        ADT_RETURN_IF_ERR(ctx->InitObjNodes(ptn_node, obj_nodes));
        inited = (obj_nodes.size() > 0);
      } else {
        ADT_LET_CONST_REF(
            obj_nodes,
            GetMatchedObjNodesFromConnected(*ctx, ptn_node, node, is_upstream));
        ADT_RETURN_IF_ERR(ctx->UpdateObjNodes(ptn_node, obj_nodes));
      }
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(ForEachInitedUpstream(*ctx, ptn_node, InitOrUpdate));
    ADT_RETURN_IF_ERR(ForEachInitedDownstream(*ctx, ptn_node, InitOrUpdate));
    ADT_CHECK(inited) << adt::errors::MismatchError{
        "ptn_node not successfully inited."};
    return adt::Ok{};
  }

  adt::Result<SmallSet<obj_node_t>> GetMatchedObjNodesFromConnected(
      const GraphMatchCtxImpl<ObjT, PtnT>& ctx,
      const ptn_node_t& ptn_node,
      const ptn_node_t& from_node,
      tIsUpstream<bool> is_from_node_upstream) {
    SmallSet<obj_node_t> obj_nodes;
    const auto& DoEachMatched =
        [&](const obj_node_t& obj_node) -> adt::Result<adt::Ok> {
      obj_nodes.insert(obj_node);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitMatchedObjNodesFromConnected(
        ctx, ptn_node, from_node, is_from_node_upstream, DoEachMatched));
    return obj_nodes;
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitMatchedObjNodesFromConnected(
      const GraphMatchCtxImpl<ObjT, PtnT>& ctx,
      const ptn_node_t& ptn_node,
      const ptn_node_t& from_node,
      tIsUpstream<bool> is_from_node_upstream,
      const DoEachT& DoEach) {
    const auto& VisitObjNode =
        [&](const obj_node_t& obj_node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(matched, ObjNodeMatch(obj_node, ptn_node));
      if (!matched) {
        return adt::Ok{};
      }
      const auto& opt_matched_ptn_node = ctx.GetMatchedPtnNode(obj_node);
      if (!opt_matched_ptn_node.has_value() ||
          opt_matched_ptn_node.value() == ptn_node) {
        return DoEach(obj_node);
      }
      return adt::Ok{};
    };
    ADT_LET_CONST_REF(from_obj_nodes_ptr, ctx.GetObjNodes(from_node));
    for (const auto& from_obj_node : *from_obj_nodes_ptr) {
      if (is_from_node_upstream.value()) {
        ADT_LET_CONST_REF(downstream_obj_nodes,
                          from_obj_node.DownstreamNodes());
        ADT_RETURN_IF_ERR(downstream_obj_nodes.VisitNodes(VisitObjNode));
      } else {
        ADT_LET_CONST_REF(upstream_obj_nodes, from_obj_node.UpstreamNodes());
        ADT_RETURN_IF_ERR(upstream_obj_nodes.VisitNodes(VisitObjNode));
      }
    }
    return adt::Ok{};
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> ForEachInitedUpstream(
      const GraphMatchCtxImpl<ObjT, PtnT>& ctx,
      const ptn_node_t& ptn_node,
      const DoEachT& DoEach) {
    ADT_LET_CONST_REF(upstreams, ptn_node.UpstreamNodes());
    return upstreams.VisitNodes([&](const auto& src) -> adt::Result<adt::Ok> {
      if (ctx.HasObjNode(src)) {
        return DoEach(src, tIsUpstream<bool>{true});
      }
      return adt::Ok{};
    });
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> ForEachInitedDownstream(
      const GraphMatchCtxImpl<ObjT, PtnT>& ctx,
      const ptn_node_t& ptn_node,
      const DoEachT& DoEach) {
    ADT_LET_CONST_REF(downstreams, ptn_node.DownstreamNodes());
    return downstreams.VisitNodes([&](const auto& dst) -> adt::Result<adt::Ok> {
      if (ctx.HasObjNode(dst)) {
        return DoEach(dst, tIsUpstream<bool>{false});
      }
      return adt::Ok{};
    });
  }

  adt::Result<bool> IgnoredPtnNode(const ptn_node_t& ptn_node) {
    ADT_LET_CONST_REF(ptn, ptn_node.Get());
    return IgnoredPtn(ptn);
  }

  adt::Result<bool> ObjNodeMatch(const obj_node_t& obj_node,
                                 const ptn_node_t& ptn_node) {
    ADT_LET_CONST_REF(obj, obj_node.Get());
    ADT_LET_CONST_REF(ptn, ptn_node.Get());
    return ObjMatch(obj, ptn);
  }

  std::function<adt::Result<bool>(const PtnT&)> IgnoredPtn;
  std::function<adt::Result<bool>(const ObjT&, const PtnT&)> ObjMatch;
};

}  // namespace ap::graph
