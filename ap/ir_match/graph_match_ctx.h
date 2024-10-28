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
  using Self = GraphMatchCtxImpl;

  using DrrValue = drr::Value;
  using DrrNode = drr::Node<DrrValue>;
  using DrrGraphNode = graph::Node<DrrNode>;
  using DrrPackedIrOp = drr::PackedIrOp<DrrValue, DrrNode>;
  using DrrNativeIrOp = drr::NativeIrOp<DrrValue, DrrNode>;
  using DrrPackedIrValue = drr::PackedIrValue<DrrNode>;
  using DrrNativeIrValue = drr::NativeIrValue<DrrNode>;
  using DrrPackedIrOpOperand = drr::PackedIrOpOperand<DrrNode>;
  using DrrNativeIrOpOperand = drr::NativeIrOpOperand<DrrNode>;
  using DrrPackedIrOpResult = drr::PackedIrOpResult<DrrNode>;
  using DrrNativeIrOpResult = drr::NativeIrOpResult<DrrNode>;
  using sg_node_t = DrrGraphNode;

  TopoMatchCtx<bg_node_t, sg_node_t> topo_match_ctx;

  bool operator==(const GraphMatchCtxImpl& other) const {
    return this == &other;
  }
  std::size_t num_matched_bg_nodes() const {
    return topo_match_ctx->num_matched_bg_nodes();
  }

  adt::Result<bool> HasBigGraphNode(const sg_node_t& node) const {
    using RetT = adt::Result<bool>;
    ADT_LET_CONST_REF(drr_node, node.Get());
    return drr_node.Match(
        [&](const DrrPackedIrOp&) -> RetT {
          return GetOptional<SGNode2BGNodesThroughConnected>(node).has_value();
        },
        [&](const DrrNativeIrOp&) -> RetT {
          return GetOptional<SGNode2BGNodesThroughConnected>(node).has_value();
        },
        [&](const DrrPackedIrValue&) -> RetT {
          return adt::errors::NotImplementedError{
              "GraphMatchCtxImpl::HasBigGraphNode does not support "
              "DrrPackedIrValue yet."};
        },
        [&](const DrrNativeIrValue&) -> RetT {
          return GetOptional<SGNode2BGNodesThroughConnected>(node).has_value();
        },
        [&](const DrrPackedIrOpOperand&) -> RetT {
          return topo_match_ctx->HasBigGraphNode(node);
        },
        [&](const DrrNativeIrOpOperand&) -> RetT {
          return topo_match_ctx->HasBigGraphNode(node);
        },
        [&](const DrrPackedIrOpResult&) -> RetT {
          return topo_match_ctx->HasBigGraphNode(node);
        },
        [&](const DrrNativeIrOpResult&) -> RetT {
          return topo_match_ctx->HasBigGraphNode(node);
        });
  }

  adt::Result<bg_node_t> GetSoleBigGraphNode(const sg_node_t& node) const {
    using RetT = adt::Result<bg_node_t>;
    ADT_LET_CONST_REF(drr_node, node.Get());
    return drr_node.Match(
        [&](const DrrPackedIrOp&) -> RetT {
          return GetSole<SGNode2BGNodesThroughConnected>(node);
        },
        [&](const DrrNativeIrOp&) -> RetT {
          return GetSole<SGNode2BGNodesThroughConnected>(node);
        },
        [&](const DrrPackedIrValue&) -> RetT {
          return adt::errors::NotImplementedError{
              "GraphMatchCtxImpl::GetSoleBigGraphNode does not support "
              "DrrPackedIrValue yet."};
        },
        [&](const DrrNativeIrValue&) -> RetT {
          return GetSole<SGNode2BGNodesThroughConnected>(node);
        },
        [&](const DrrPackedIrOpOperand&) -> RetT {
          return topo_match_ctx->GetSoleBigGraphNode(node);
        },
        [&](const DrrNativeIrOpOperand&) -> RetT {
          return topo_match_ctx->GetSoleBigGraphNode(node);
        },
        [&](const DrrPackedIrOpResult&) -> RetT {
          return topo_match_ctx->GetSoleBigGraphNode(node);
        },
        [&](const DrrNativeIrOpResult&) -> RetT {
          return topo_match_ctx->GetSoleBigGraphNode(node);
        });
  }

  adt::Result<std::optional<sg_node_t>> GetMatchedSmallGraphOpNode(
      const bg_node_t& bg_node) const {
    using GraphDescriptor =
        graph::GraphDescriptor<bg_node_t, drr::topo_kind::Default>;
    ADT_LET_CONST_REF(is_op_node, GraphDescriptor{}.IsOpNode(bg_node));
    ADT_CHECK(is_op_node);
    return MonoidToResultOptional<BGNode2SNodesThroughConnected>(bg_node);
  }

  template <typename MonoidT>
  adt::Result<typename MonoidT::output_type> GetSole(
      const typename MonoidT::input_type& node) const {
    ADT_LET_CONST_REF(opt_sole, MonoidToResultOptional<MonoidT>(node));
    ADT_CHECK(opt_sole.has_value());
    return opt_sole.value();
  }

  template <typename MonoidT>
  std::optional<typename MonoidT::output_type> GetOptional(
      const typename MonoidT::input_type& node) const {
    const auto& result_opt = MonoidToResultOptional<MonoidT>(node);
    if (result_opt.HasError()) {
      return std::nullopt;
    }
    return result_opt.GetOkValue();
  }

  template <typename MonoidT>
  adt::Result<std::optional<typename MonoidT::output_type>>
  MonoidToResultOptional(const typename MonoidT::input_type& node) const {
    std::optional<typename MonoidT::output_type> ret{};
    auto DoEach = [&](const typename MonoidT::output_type& output)
        -> adt::Result<adt::Ok> {
      if (ret.has_value()) {
        ADT_CHECK(ret.value() == output);
      } else {
        ret = output;
      }
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(MonoidT::Call(this, node, DoEach));
    return ret;
  }

  template <typename T>
  using DoEachT = std::function<adt::Result<adt::Ok>(const T&)>;

  struct SGNode2Downstreams {
    using input_type = sg_node_t;
    using output_type = sg_node_t;

    static adt::Result<adt::Ok> Call(const Self* self,
                                     const input_type& input,
                                     const DoEachT<output_type>& DoEach) {
      using GraphDescriptor =
          graph::GraphDescriptor<input_type, drr::topo_kind::Default>;
      return GraphDescriptor{}.VisitDownstreamNodes(input, DoEach);
    }
  };

  struct SGNode2Upstreams {
    using input_type = sg_node_t;
    using output_type = sg_node_t;

    static adt::Result<adt::Ok> Call(const Self* self,
                                     const input_type& input,
                                     const DoEachT<output_type>& DoEach) {
      using GraphDescriptor =
          graph::GraphDescriptor<input_type, drr::topo_kind::Default>;
      return GraphDescriptor{}.VisitUpstreamNodes(input, DoEach);
    }
  };

  struct SGNode2BGNodes {
    using input_type = sg_node_t;
    using output_type = bg_node_t;

    static adt::Result<adt::Ok> Call(const Self* self,
                                     const input_type& input,
                                     const DoEachT<output_type>& DoEach) {
      ADT_CHECK(self->topo_match_ctx->HasBigGraphNode(input));
      ADT_LET_CONST_REF(bg_node,
                        self->topo_match_ctx->GetSoleBigGraphNode(input));
      return DoEach(bg_node);
    }
  };

  struct BGNode2Downstreams {
    using input_type = bg_node_t;
    using output_type = bg_node_t;

    static adt::Result<adt::Ok> Call(const Self* self,
                                     const input_type& input,
                                     const DoEachT<output_type>& DoEach) {
      using GraphDescriptor =
          graph::GraphDescriptor<input_type, drr::topo_kind::Default>;
      return GraphDescriptor{}.VisitDownstreamNodes(input, DoEach);
    }
  };

  struct BGNode2Upstreams {
    using input_type = bg_node_t;
    using output_type = bg_node_t;

    static adt::Result<adt::Ok> Call(const Self* self,
                                     const input_type& input,
                                     const DoEachT<output_type>& DoEach) {
      using GraphDescriptor =
          graph::GraphDescriptor<input_type, drr::topo_kind::Default>;
      return GraphDescriptor{}.VisitUpstreamNodes(input, DoEach);
    }
  };

  struct BGNode2SGNodes {
    using input_type = bg_node_t;
    using output_type = sg_node_t;

    static adt::Result<adt::Ok> Call(const Self* self,
                                     const input_type& input,
                                     const DoEachT<output_type>& DoEach) {
      const auto& sg_node =
          self->topo_match_ctx->GetMatchedSmallGraphNode(input);
      ADT_CHECK(sg_node.has_value());
      return DoEach(sg_node.value());
    }
  };

  template <typename F0, typename F1, typename F2>
  struct Product {
    using input_type = typename F0::input_type;
    using output_type = typename F2::output_type;
    static_assert(
        std::is_same_v<typename F0::output_type, typename F1::input_type>, "");
    static_assert(
        std::is_same_v<typename F1::output_type, typename F2::input_type>, "");

    static adt::Result<adt::Ok> Call(const Self* self,
                                     const input_type& input,
                                     const DoEachT<output_type>& DoEach) {
      auto F1DoEach =
          [&](const typename F1::output_type& node) -> adt::Result<adt::Ok> {
        return F2::Call(self, node, DoEach);
      };
      auto F0DoEach =
          [&](const typename F0::output_type& node) -> adt::Result<adt::Ok> {
        return F1::Call(self, node, F1DoEach);
      };
      return F0::Call(self, input, F0DoEach);
    }
  };

  template <typename F0, typename F1>
  struct Sum {
    using input_type = typename F0::input_type;
    using output_type = typename F0::output_type;
    static_assert(
        std::is_same_v<typename F0::input_type, typename F1::input_type>, "");
    static_assert(
        std::is_same_v<typename F0::output_type, typename F1::output_type>, "");

    static adt::Result<adt::Ok> Call(const Self* self,
                                     const input_type& input,
                                     const DoEachT<output_type>& DoEach) {
      ADT_RETURN_IF_ERR(F0::Call(self, input, DoEach));
      ADT_RETURN_IF_ERR(F1::Call(self, input, DoEach));
      return adt::Ok{};
    }
  };

  struct SGNode2BGNodesThroughConnected
      : public Sum<
            Product<SGNode2Upstreams, SGNode2BGNodes, BGNode2Downstreams>,
            Product<SGNode2Downstreams, SGNode2BGNodes, BGNode2Upstreams>> {};

  struct BGNode2SNodesThroughConnected
      : public Sum<
            Product<BGNode2Downstreams, BGNode2SGNodes, SGNode2Upstreams>,
            Product<BGNode2Upstreams, BGNode2SGNodes, SGNode2Downstreams>> {};
};

template <typename bg_node_t /*big graph node type*/>
DEFINE_ADT_RC(GraphMatchCtx, GraphMatchCtxImpl<bg_node_t>);

}  // namespace ap::ir_match
