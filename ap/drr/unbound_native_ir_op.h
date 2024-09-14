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
#include "ap/drr/native_ir_op_declare.h"
#include "ap/drr/tags.h"
#include "ap/graph/node.h"

namespace ap::drr {

template <typename ValueT, typename NodeT>
struct UnboundNativeIrOpImpl {
  NativeIrOpDeclare<ValueT, NodeT> op_declare;
  std::string name;

  bool operator==(const UnboundNativeIrOpImpl& other) const {
    return this->op_declare == other.op_declare && this->name == other.name;
  }
};

template <typename ValueT, typename NodeT>
DEFINE_ADT_RC(UnboundNativeIrOp, UnboundNativeIrOpImpl<ValueT, NodeT>);

}  // namespace ap::drr

namespace ap::axpr {

template <typename ValueT, typename NodeT>
struct TypeImpl<drr::tSrcPtn<drr::UnboundNativeIrOp<ValueT, NodeT>>>
    : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "SrcPtnUnboundNativeIrOp"; }
};

template <typename ValueT, typename NodeT>
struct TypeImpl<drr::tResPtn<drr::UnboundNativeIrOp<ValueT, NodeT>>>
    : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "ResPtnUnboundNativeIrOp"; }
};

}  // namespace ap::axpr
