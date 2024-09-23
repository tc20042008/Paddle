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
#include "ap/graph/node_cstr.h"

namespace ap::drr {

template <typename NodeT>
struct NativeIrOpResultImpl {
  graph::Node<NodeT> node;
  std::size_t index;

  bool operator==(const NativeIrOpResultImpl& other) const {
    return this->node == other.name && this->index == other.index;
  }

  graph::NativeIrOpResultCstr node_cstr() const {
    return graph::NativeIrOpResultCstr{index};
  }
};

template <typename NodeT>
DEFINE_ADT_RC(NativeIrOpResult, NativeIrOpResultImpl<NodeT>);

}  // namespace ap::drr
