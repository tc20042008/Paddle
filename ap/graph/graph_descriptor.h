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
#include "ap/graph/node_cstr.h"
#include "glog/logging.h"

namespace ap::graph {

template <typename NodeT>
struct GraphDescriptor {
  GraphDescriptor(const GraphDescriptor&) = default;
  GraphDescriptor(GraphDescriptor&&) = default;

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitAllNodes(const DoEachT& DoEach) const;

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitUpstreamNodes(const NodeT&,
                                          const DoEachT& DoEach) const;

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitDownstreamNodes(const NodeT&,
                                            const DoEachT& DoEach) const;

  adt::Result<graph::NodeCstr> GetNodeConstraint(const NodeT&) const;

  adt::Result<bool> IgnoredNode(const NodeT&) const;

  adt::Result<bool> Satisfy(const NodeT&, const NodeCstr&) const;
};

}  // namespace ap::graph
