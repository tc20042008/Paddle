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
#include "ap/axpr/adt.h"
#include "ap/drr/native_ir_op.h"
#include "ap/drr/native_ir_op_operand.h"
#include "ap/drr/native_ir_op_result.h"
#include "ap/drr/native_ir_value.h"
#include "ap/drr/packed_ir_op.h"
#include "ap/drr/packed_ir_op_operand.h"
#include "ap/drr/packed_ir_op_result.h"
#include "ap/drr/packed_ir_value.h"

namespace ap::drr {

template <typename ValueT, typename NodeT>
using NodeImpl = std::variant<NativeIrValue<NodeT>,
                              NativeIrOp<ValueT, NodeT>,
                              NativeIrOpOperand<NodeT>,
                              NativeIrOpResult<NodeT>,
                              PackedIrValue<NodeT>,
                              PackedIrOp<ValueT, NodeT>,
                              PackedIrOpOperand<NodeT>,
                              PackedIrOpResult<NodeT>>;

template <typename ValueT>
struct Node : public NodeImpl<ValueT, Node<ValueT>> {
  using NodeImpl<ValueT, Node<ValueT>>::NodeImpl;
  DEFINE_ADT_VARIANT_METHODS(NodeImpl<ValueT, Node<ValueT>>);

  const graph::Node<Node<ValueT>>& node() const {
    return Match([](const auto& impl) -> const graph::Node<Node<ValueT>>& {
      return impl->node;
    });
  }
};

}  // namespace ap::drr
