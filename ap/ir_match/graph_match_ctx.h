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
#include "ap/graph/node_descriptor.h"

namespace ap::ir_match {

template <typename bg_node_t /*big graph node type*/,
          typename sg_node_t /*small graph node type*/>
struct GraphMatchCtxImpl {
  GraphMatchCtxImpl() {}
  GraphMatchCtxImpl(const GraphMatchCtxImpl&) = default;
  GraphMatchCtxImpl(GraphMatchCtxImpl&&) = default;

  bool HasObjNode(const sg_node_t& node) const {
    return this->sg_node2bg_nodes_.count(node) > 0;
  }

  adt::Result<bg_node_t> GetSoleBigGraphNode(const sg_node_t& node) const {
    ADT_LET_CONST_REF(bg_nodes, GetObjNodes(node));
    ADT_CHECK(bg_nodes->size(), 1);
    return *bg_nodes->begin();
  }

  adt::Result<const std::unordered_set<bg_node_t>*> GetObjNodes(
      const sg_node_t& node) const {
    const auto& iter = this->sg_node2bg_nodes_.find(node);
    if (iter == this->sg_node2bg_nodes_.end()) {
      return adt::errors::KeyError{
          std::string() + "no node_id " +
          graph::NodeDescriptor<sg_node_t>{}.DebugId(node) + " found."};
    }
    return &iter->second;
  }

  std::optional<sg_node_t> GetMatchedPtnNode(const bg_node_t& bg_node) const {
    const auto& iter = matched_bg_node2sg_node_.find(bg_node);
    if (iter == matched_bg_node2sg_node_.end()) {
      return std::nullopt;
    }
    return iter->second;
  }

  adt::Result<adt::Ok> InitObjNodes(const sg_node_t& sg_node,
                                    const std::unordered_set<bg_node_t>& val) {
    VLOG(0) << "InitObjNodes. sg_node: "
            << graph::NodeDescriptor<sg_node_t>{}.DebugId(sg_node)
            << ", val:" <<
        [&] {
          std::ostringstream ss;
          for (const auto& val_node : val) {
            ss << graph::NodeDescriptor<bg_node_t>{}.DebugId(val_node) << " ";
          }
          return ss.str();
        }();
    auto* ptr = &this->sg_node2bg_nodes_[sg_node];
    ADT_CHECK(ptr->empty()) << adt::errors::KeyError{
        "InitObjNodes failed. 'sg_node' has been matched to existed "
        "bg_nodes"};
    ADT_CHECK(!val.empty()) << adt::errors::MismatchError{
        "GraphMatchCtxImpl::InitObjNodes: sg_node should not be matched to "
        "empty."};
    for (const auto& bg_node : val) {
      ADT_CHECK(!GetMatchedPtnNode(bg_node).has_value())
          << adt::errors::KeyError{
                 "GraphMatchCtxImpl::InitObjNodes failed. there is matched "
                 "bg_node in 'val'"};
    }
    *ptr = val;
    if (ptr->size() == 1) {
      ADT_CHECK(matched_bg_node2sg_node_.emplace(*val.begin(), sg_node).second);
    }
    return adt::Ok{};
  }

  adt::Result<adt::Ok> UpdateObjNodes(
      const sg_node_t& sg_node, const std::unordered_set<bg_node_t>& val) {
    ADT_CHECK(!val.empty());
    for (const auto& bg_node : val) {
      const auto& opt_matched = GetMatchedPtnNode(bg_node);
      ADT_CHECK(!opt_matched.has_value() || opt_matched.value() == sg_node)
          << adt::errors::KeyError{
                 "UpdateObjNodes failed. there is matched bg_node in 'val'"};
    }
    std::unordered_set<bg_node_t> intersection;
    auto* ptr = &this->sg_node2bg_nodes_[sg_node];
    for (const auto& lhs : *ptr) {
      if (val.count(lhs) > 0) {
        intersection.insert(lhs);
      }
    }
    VLOG(0) << "UpdateObjNodes. sg_node: "
            << graph::NodeDescriptor<sg_node_t>{}.DebugId(sg_node)
            << ", old_val:" <<
        [&] {
          std::ostringstream ss;
          for (const auto& val_node : *ptr) {
            ss << graph::NodeDescriptor<bg_node_t>{}.DebugId(val_node) << " ";
          }
          return ss.str();
        }()
            << ", new_val:" <<
        [&] {
          std::ostringstream ss;
          for (const auto& val_node : val) {
            ss << graph::NodeDescriptor<bg_node_t>{}.DebugId(val_node) << " ";
          }
          return ss.str();
        }()
            << ", intersection: " <<
        [&] {
          std::ostringstream ss;
          for (const auto& val_node : intersection) {
            ss << graph::NodeDescriptor<bg_node_t>{}.DebugId(val_node) << " ";
          }
          return ss.str();
        }();
    *ptr = intersection;
    ADT_CHECK(!ptr->empty()) << adt::errors::MismatchError{
        "GraphMatchCtxImpl::UpdateObjNodes: intersection is empty."};
    if (ptr->size() == 1) {
      const auto& iter =
          matched_bg_node2sg_node_.emplace(*ptr->begin(), sg_node).first;
      ADT_CHECK(iter->second == sg_node);
    }
    return adt::Ok{};
  }

  bool operator==(const GraphMatchCtxImpl& other) const {
    return this == &other;
  }

  std::size_t num_matched_bg_nodes() const {
    return matched_bg_node2sg_node_.size();
  }

 private:
  std::unordered_map<sg_node_t, std::unordered_set<bg_node_t>>
      sg_node2bg_nodes_;
  std::unordered_map<bg_node_t, sg_node_t> matched_bg_node2sg_node_;
};

template <typename bg_node_t /*big graph node type*/,
          typename sg_node_t /*small graph node type*/>
DEFINE_ADT_RC(GraphMatchCtx, GraphMatchCtxImpl<bg_node_t, sg_node_t>);

}  // namespace ap::ir_match
