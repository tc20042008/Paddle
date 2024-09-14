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
#include "ap/graph/node.h"
#include "ap/graph/tags.h"

namespace ap::graph {

template <typename T>
class NodeArena : public std::enable_shared_from_this<NodeArena<T>> {
 public:
  NodeArena() {}
  NodeArena(const NodeArena&) = delete;
  NodeArena(NodeArena&&) = delete;

  adt::Result<T> At(const tNodeId<size_t>& node_id) const {
    if (node_id.value() >= nodes_.size()) {
      return adt::errors::IndexError{"node_id out of ranges."};
    }
    return nodes_.at(node_id.value());
  }

  template <typename ConstructorT>
  const T& New(const ConstructorT& Constructor) {
    tNodeId<size_t> node_id{nodes_.size()};
    nodes_.emplace_back(
        Constructor(Node<T>{node_id, this->shared_from_this()}));
    return nodes_.at(nodes_.size() - 1);
  }

  template <typename ConstructorT>
  adt::Result<T> TryNew(const ConstructorT& Constructor) {
    tNodeId<size_t> node_id{nodes_.size()};
    ADT_LET_CONST_REF(node,
                      Constructor(Node<T>{node_id, this->shared_from_this()}));
    nodes_.emplace_back(node);
    return nodes_.at(nodes_.size() - 1);
  }

  adt::Result<std::optional<adt::List<Node<T>>>> DstNodes4SrcNodeId(
      const tNodeId<size_t>& src_id) {
    if (src_id >= src_node_id2dst_nodes.size()) {
      return adt::errors::IndexError{"src node_id out of ranges."};
    }
    return src_node_id2dst_nodes.at(src_id);
  }

  adt::Result<std::optional<adt::List<Node<T>>>> SrcNodes4DstNodeId(
      const tNodeId<size_t>& dst_id) {
    if (dst_id >= dst_node_id2src_nodes.size()) {
      return adt::errors::IndexError{"dst node_id out of ranges."};
    }
    return dst_node_id2src_nodes.at(dst_id);
  }

  adt::Result<adt::Ok> Connect(const Node<T>& src_node,
                               const Node<T>& dst_node) {
    const auto& src_id = src_node.node_id();
    if (src_node.node_arena().lock() != this->shared_from_this()) {
      return adt::errors::RuntimeError{
          "Connection between nodes from different arena is not supported. "};
    }
    if (src_id.value() >= src_node_id2dst_nodes.size()) {
      src_node_id2dst_nodes.resize(src_id.value() + 1);
    }
    auto* dst_nodes_ptr = &src_node_id2dst_nodes[src_id.value()];
    if (!dst_nodes_ptr->has_value()) {
      *dst_nodes_ptr = adt::List<Node<T>>{};
    }
    const auto& dst_id = dst_node.node_id();
    dst_nodes_ptr->value()->emplace_back(
        Node<T>{dst_id, this->shared_from_this()});
    if (dst_node.node_arena().lock() != this->shared_from_this()) {
      return adt::errors::RuntimeError{
          "Connection between nodes from different arena is not supported. "};
    }
    if (dst_id.value() >= dst_node_id2src_nodes.size()) {
      dst_node_id2src_nodes.resize(dst_id.value() + 1);
    }
    auto* src_nodes_ptr = &dst_node_id2src_nodes[dst_id.value()];
    if (!src_nodes_ptr->has_value()) {
      *src_nodes_ptr = adt::List<Node<T>>{};
    }
    src_nodes_ptr->value()->emplace_back(
        Node<T>{src_id, this->shared_from_this()});
    return adt::Ok{};
  }

 private:
  std::vector<T> nodes_;
  std::vector<std::optional<adt::List<Node<T>>>> src_node_id2dst_nodes;
  std::vector<std::optional<adt::List<Node<T>>>> dst_node_id2src_nodes;
};

template <typename T>
adt::Result<T> Node<T>::Get() const {
  ADT_LET_CONST_REF(arena, adt::WeakPtrLock(this->node_arena()));
  return arena->At(this->node_id());
}

template <typename T>
adt::Result<std::optional<adt::List<Node<T>>>> Node<T>::DownstreamNodes()
    const {
  ADT_LET_CONST_REF(arena, adt::WeakPtrLock(this->node_arena()));
  return arena->DstNodes4SrcNodeId(this->node_id());
}

template <typename T>
adt::Result<std::optional<adt::List<Node<T>>>> Node<T>::UpStreamNodes() const {
  ADT_LET_CONST_REF(arena, adt::WeakPtrLock(this->node_arena()));
  return arena->SrcNodes4DstNodeId(this->node_id());
}

template <typename T>
adt::Result<adt::Ok> Node<T>::ConnectTo(const Node& dst_node) const {
  ADT_LET_CONST_REF(arena, adt::WeakPtrLock(this->node_arena()));
  return arena->Connect(*this, dst_node);
}

}  // namespace ap::graph
