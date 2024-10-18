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
#include "ap/drr/drr_value.h"
#include "ap/drr/node.h"
#include "ap/drr/source_pattern_ctx.h"
#include "ap/graph/node.h"
#include "ap/ir_match/graph_match_ctx.h"
#include "ap/ir_match/op_match_ctx.h"
#include "ap/ir_match/tensor_match_ctx.h"

namespace ap::ir_match {

template <typename IrNodeT>
struct IrMatchCtxImpl {
  using DrrNodeT = drr::Node<drr::Value>;
  using SmallGraphNodeT = graph::Node<DrrNodeT>;
  drr::SourcePatternCtx<drr::Value, DrrNodeT> source_pattern_ctx;
  GraphMatchCtx<IrNodeT, SmallGraphNodeT> graph_match_ctx;
};

template <typename IrNodeT>
DEFINE_ADT_RC(IrMatchCtx, IrMatchCtxImpl<IrNodeT>);

}  // namespace ap::ir_match

namespace ap::axpr {

template <typename IrNodeT>
struct TypeImpl<ir_match::IrMatchCtx<IrNodeT>> : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "IrMatchCtx"; }
};

}  // namespace ap::axpr
