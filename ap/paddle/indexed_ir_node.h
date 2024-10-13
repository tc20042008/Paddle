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
#include "ap/index_expr/index_tuple_expr.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"

namespace ap::paddle {

template <typename NodeT>
struct IndexedIrValueImpl {
  graph::Node<NodeT> node;
  pir::Value value;
  index_expr::IndexTupleExpr indexes_expr;

  bool operator==(const IndexedIrValueImpl& other) const {
    return this->value == other.value &&
           this->indexes_expr == other.indexes_expr;
  }
};

template <typename NodeT>
DEFINE_ADT_RC(IndexedIrValue, IndexedIrValueImpl<NodeT>);

template <typename NodeT>
struct IndexedIrOpImpl {
  graph::Node<NodeT> node;
  pir::Operation* op;

  bool operator==(const IndexedIrOpImpl& other) const {
    return this->op == other.op;
  }
};

template <typename NodeT>
DEFINE_ADT_RC(IndexedIrOp, IndexedIrOpImpl<NodeT>);

template <typename NodeT>
using IndexedIrNodeImpl =
    std::variant<IndexedIrValue<NodeT>, IndexedIrOp<NodeT>>;

struct IndexedIrNode : public IndexedIrNodeImpl<IndexedIrNode> {
  using IndexedIrNodeImpl<IndexedIrNode>::IndexedIrNodeImpl;

  DEFINE_ADT_VARIANT_METHODS(IndexedIrNodeImpl<IndexedIrNode>);

  const graph::Node<IndexedIrNode>& node() const {
    return Match([](const auto& impl) -> const graph::Node<IndexedIrNode>& {
      return impl->node;
    });
  }
};

}  // namespace ap::paddle
