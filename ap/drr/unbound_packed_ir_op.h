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
#include "ap/axpr/type.h"
#include "ap/drr/packed_ir_op_declare.h"
#include "ap/drr/tags.h"
#include "ap/graph/node.h"

namespace ap::drr {

template <typename ValueT, typename NodeT>
struct UnboundPackedIrOpImpl {
 public:
  PackedIrOpDeclare<ValueT, NodeT> op_declare;
  std::string name;
  bool operator==(const UnboundPackedIrOpImpl& other) const {
    return this->op_declare == other.op_declare && this->name == other.name;
  }
};

template <typename ValueT, typename NodeT>
DEFINE_ADT_RC(UnboundPackedIrOp, UnboundPackedIrOpImpl<ValueT, NodeT>);

}  // namespace ap::drr

namespace ap::axpr {

template <typename ValueT, typename NodeT>
struct TypeImpl<drr::tSrcPtn<drr::UnboundPackedIrOp<ValueT, NodeT>>>
    : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "SrcPtnUnboundPackedIrOp"; }
};

template <typename ValueT, typename NodeT>
struct TypeImpl<drr::tResPtn<drr::UnboundPackedIrOp<ValueT, NodeT>>>
    : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "ResPtnUnboundPackedIrOp"; }
};

}  // namespace ap::axpr
