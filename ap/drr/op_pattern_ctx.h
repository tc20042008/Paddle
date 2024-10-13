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

#include <map>
#include "ap/adt/adt.h"
#include "ap/drr/ir_op.h"
#include "ap/drr/tags.h"
#include "ap/graph/node_arena.h"
#include "ap/graph/tags.h"

namespace ap::drr {

template <typename ValueT, typename NodeT>
struct DrrCtxImpl;

template <typename ValueT, typename NodeT>
struct OpPatternCtxImpl {
  std::shared_ptr<graph::NodeArena<NodeT>> node_arena;
  mutable std::map<std::string, IrOp<ValueT, NodeT>> uid2ir_op;
  std::weak_ptr<DrrCtxImpl<ValueT, NodeT>> drr_ctx;

  bool operator==(const OpPatternCtxImpl& other) const {
    return this == &other;
  }
};

template <typename ValueT, typename NodeT>
DEFINE_ADT_RC(OpPatternCtx, OpPatternCtxImpl<ValueT, NodeT>);

}  // namespace ap::drr

namespace ap::axpr {

template <typename ValueT, typename NodeT>
struct TypeImpl<drr::tSrcPtn<drr::OpPatternCtx<ValueT, NodeT>>>
    : public std::monostate {
  using value_type = drr::tSrcPtn<drr::OpPatternCtx<ValueT, NodeT>>;

  const char* Name() const { return "SrcPtnOpPatternCtx"; }
};

template <typename ValueT, typename NodeT>
struct TypeImpl<drr::tResPtn<drr::OpPatternCtx<ValueT, NodeT>>>
    : public std::monostate {
  using value_type = drr::tResPtn<drr::OpPatternCtx<ValueT, NodeT>>;

  const char* Name() const { return "ResPtnOpPatternCtx"; }
};

}  // namespace ap::axpr
