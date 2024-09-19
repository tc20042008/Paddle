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

namespace ap::drr {

template <typename ValueT>
adt::Result<bool> IsIgnoredPackedNode(const Node<ValueT>& node) {
  using NodeT = Node<ValueT>;
  return node.Match(
      [](const PackedIrValue<NodeT>&) -> adt::Result<bool> { return true; },
      [&](const PackedIrOpOperand<NodeT>& impl) -> adt::Result<bool> {
        ADT_LET_CONST_REF(upstreams, impl->node.UpstreamNodes());
        ADT_CHECK(upstreams.size(), 1);
        ADT_LET_CONST_REF(upstream_node, upstreams.Sole());
        ADT_LET_CONST_REF(upstream, upstream_node.Get());
        return IsIgnoredPackedNode(upstream);
      },
      [&](const PackedIrOpResult<NodeT>& impl) -> adt::Result<bool> {
        ADT_LET_CONST_REF(downstreams, impl->node.DownstreamNodes());
        ADT_CHECK(downstreams.size(), 1);
        ADT_LET_CONST_REF(downstream_node, downstreams.Sole());
        ADT_LET_CONST_REF(downstream, downstream_node.Get());
        return IsIgnoredPackedNode(downstream);
      },
      [](const auto&) -> adt::Result<bool> { return false; });
}

template <typename ValueT>
adt::Result<bool> IsObjMatch(const Node<ValueT>& obj, const Node<ValueT>& ptn) {
  ADT_LET_CONST_REF(ptn_ignored, IsIgnoredPackedNode(ptn));
  ADT_CHECK(!ptn_ignored);
  ADT_LET_CONST_REF(obj_ignored, IsIgnoredPackedNode(obj));
  if (obj_ignored) {
    return false;
  }
  using NodeT = Node<ValueT>;
  using NativeIrValueT = NativeIrValue<NodeT>;
  using NativeIrOpT = NativeIrOp<ValueT, NodeT>;
  using NativeIrOpOperandT = NativeIrOpOperand<NodeT>;
  using NativeIrOpResultT = NativeIrOpResult<NodeT>;
  using PackedIrOpT = PackedIrOp<ValueT, NodeT>;
  using PackedIrOpOperandT = PackedIrOpOperand<NodeT>;
  using PackedIrOpResultT = PackedIrOpResult<NodeT>;
  const auto& pattern_match = ::common::Overloaded{
      [](const NativeIrValueT&, const NativeIrValueT&) -> adt::Result<bool> {
        return true;
      },
      [](const NativeIrOpT& lhs, const NativeIrOpT& rhs) -> adt::Result<bool> {
        return lhs->op_declare->op_name == rhs->op_declare->op_name;
      },
      [](const NativeIrOpOperandT& lhs, const NativeIrOpOperandT& rhs)
          -> adt::Result<bool> { return lhs->index == rhs->index; },
      [](const NativeIrOpResultT& lhs, const NativeIrOpResultT& rhs)
          -> adt::Result<bool> { return lhs->index == rhs->index; },
      [](const PackedIrOpT& lhs, const PackedIrOpT& rhs) -> adt::Result<bool> {
        return lhs->op_declare->op_name == rhs->op_declare->op_name;
      },
      [](const PackedIrOpOperandT& lhs,
         const PackedIrOpOperandT& rhs) -> adt::Result<bool> { return true; },
      [](const PackedIrOpResultT& lhs,
         const PackedIrOpResultT& rhs) -> adt::Result<bool> { return true; },
      [](const auto&, const auto&) -> adt::Result<bool> { return false; }};
  return std::visit(pattern_match, obj.variant(), ptn.variant());
}

}  // namespace ap::drr
