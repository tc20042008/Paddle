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

#include <sstream>
#include <unordered_map>
#include "ap/graph/adt.h"
#include "ap/graph/node.h"

namespace ap::graph {

template <typename ObjectGraphNodeT, typename PatternGraphNodeT>
struct GraphMatchCtxImpl {
  GraphMatchCtxImpl() {}
  GraphMatchCtxImpl(const GraphMatchCtxImpl&) = default;
  GraphMatchCtxImpl(GraphMatchCtxImpl&&) = default;

  using obj_node_t = graph::Node<ObjectGraphNodeT>;
  using ptn_node_t = graph::Node<PatternGraphNodeT>;

  bool HasObjNode(const ptn_node_t& node) const {
    return this->ptn_node2obj_nodes_.count(node) > 0;
  }

  adt::Result<const SmallSet<obj_node_t>*> GetObjNodes(
      const ptn_node_t& node) const {
    const auto& iter = this->ptn_node2obj_nodes_.find(node);
    if (iter == this->ptn_node2obj_nodes_.end()) {
      return adt::errors::KeyError{std::string() + "no node_id " +
                                   std::to_string(node.node_id().value()) +
                                   " found."};
    }
    return &iter->second;
  }

  std::optional<ptn_node_t> GetMatchedPtnNode(
      const obj_node_t& obj_node) const {
    const auto& iter = matched_obj_node2ptn_node_.find(obj_node);
    if (iter == matched_obj_node2ptn_node_.end()) {
      return std::nullopt;
    }
    return iter->second;
  }

  adt::Result<adt::Ok> InitObjNodes(const ptn_node_t& ptn_node,
                                    const SmallSet<obj_node_t>& val) {
    VLOG(10) << "InitObjNodes. ptn_node: " << ptn_node.node_id().value()
             << ", val:" << [&] {
                  std::ostringstream ss;
                  for (const auto& val_node : val) {
                    ss << val_node.node_id().value() << " ";
                  }
                  return ss.str();
                }();
    auto* ptr = &this->ptn_node2obj_nodes_[ptn_node];
    ADT_CHECK(ptr->empty()) << adt::errors::KeyError{
        "InitObjNodes failed. 'ptn_node' has been matched to existed "
        "obj_nodes"};
    ADT_CHECK(!val.empty()) << adt::errors::MismatchError{
        "GraphMatchCtxImpl::InitObjNodes: ptn_node should not be matched to "
        "empty."};
    for (const auto& obj_node : val) {
      ADT_CHECK(!GetMatchedPtnNode(obj_node).has_value())
          << adt::errors::KeyError{
                 "GraphMatchCtxImpl::InitObjNodes failed. there is matched "
                 "obj_node in 'val'"};
    }
    *ptr = val;
    if (ptr->size() == 1) {
      ADT_CHECK(
          matched_obj_node2ptn_node_.emplace(*val.begin(), ptn_node).second);
    }
    return adt::Ok{};
  }

  adt::Result<adt::Ok> UpdateObjNodes(const ptn_node_t& ptn_node,
                                      const SmallSet<obj_node_t>& val) {
    for (const auto& obj_node : val) {
      const auto& opt_matched = GetMatchedPtnNode(obj_node);
      ADT_CHECK(!opt_matched.has_value() || opt_matched.value() == ptn_node)
          << adt::errors::KeyError{
                 "UpdateObjNodes failed. there is matched obj_node in 'val'"};
    }
    SmallSet<obj_node_t> intersection;
    auto* ptr = &this->ptn_node2obj_nodes_[ptn_node];
    for (const auto& lhs : *ptr) {
      if (val.count(lhs) > 0) {
        intersection.insert(lhs);
      }
    }
    VLOG(10) << "UpdateObjNodes. ptn_node: " << ptn_node.node_id().value()
             << ", old_val:" <<
        [&] {
          std::ostringstream ss;
          for (const auto& val_node : *ptr) {
            ss << val_node.node_id().value() << " ";
          }
          return ss.str();
        }()
             << ", new_val:" <<
        [&] {
          std::ostringstream ss;
          for (const auto& val_node : val) {
            ss << val_node.node_id().value() << " ";
          }
          return ss.str();
        }()
             << ", intersection: " << [&] {
                  std::ostringstream ss;
                  for (const auto& val_node : intersection) {
                    ss << val_node.node_id().value() << " ";
                  }
                  return ss.str();
                }();
    *ptr = intersection;
    ADT_CHECK(!ptr->empty()) << adt::errors::MismatchError{
        "GraphMatchCtxImpl::UpdateObjNodes: intersection is empty."};
    if (ptr->size() == 1) {
      const auto& iter =
          matched_obj_node2ptn_node_.emplace(*ptr->begin(), ptn_node).first;
      ADT_CHECK(iter->second == ptn_node);
    }
    return adt::Ok{};
  }

  bool operator==(const GraphMatchCtxImpl& other) const {
    return this == &other;
  }

  std::size_t num_matched_obj_nodes() const {
    return matched_obj_node2ptn_node_.size();
  }

 private:
  std::unordered_map<ptn_node_t, SmallSet<obj_node_t>> ptn_node2obj_nodes_;
  std::unordered_map<obj_node_t, ptn_node_t> matched_obj_node2ptn_node_;
};

template <typename ObjectGraphNodeT, typename PatternGraphNodeT>
DEFINE_ADT_RC(GraphMatchCtx,
              GraphMatchCtxImpl<ObjectGraphNodeT, PatternGraphNodeT>);

}  // namespace ap::graph
