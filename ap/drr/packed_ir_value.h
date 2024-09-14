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
#include "ap/drr/tags.h"
#include "ap/graph/node.h"

namespace ap::drr {

template <typename NodeT>
struct PackedIrValueImpl {
  graph::Node<NodeT> node;
  std::string name;

  bool operator==(const PackedIrValueImpl& other) const {
    return this->node == other.node && this->name == other.name;
  }
};

template <typename NodeT>
DEFINE_ADT_RC(PackedIrValue, PackedIrValueImpl<NodeT>);

}  // namespace ap::drr

namespace ap::axpr {

template <typename NodeT>
struct TypeImpl<drr::tSrcPtn<drr::PackedIrValue<NodeT>>>
    : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "SrcPtnPackedIrValue"; }
};

template <typename NodeT>
struct TypeImpl<drr::tStarred<drr::tSrcPtn<drr::PackedIrValue<NodeT>>>>
    : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "StarredSrcPtnPackedIrValue"; }
};

template <typename NodeT>
struct TypeImpl<drr::tResPtn<drr::PackedIrValue<NodeT>>>
    : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "ResPtnPackedIrValue"; }
};

template <typename NodeT>
struct TypeImpl<drr::tStarred<drr::tResPtn<drr::PackedIrValue<NodeT>>>>
    : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "StarredResPtnPackedIrValue"; }
};

}  // namespace ap::axpr
