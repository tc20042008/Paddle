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
#include "ap/drr/result_pattern_ctx.h"
#include "ap/drr/source_pattern_ctx.h"
#include "ap/drr/tags.h"

namespace ap::drr {

template <typename ValueT, typename NodeT>
struct DrrCtxImpl {
  std::shared_ptr<graph::NodeArena<NodeT>> node_arena;
  std::optional<std::string> pass_name;
  std::optional<SourcePatternCtx<ValueT, NodeT>> source_pattern_ctx;
  std::optional<ResultPatternCtx<ValueT, NodeT>> result_pattern_ctx;

  bool operator==(const DrrCtxImpl& other) const { return this == &other; }
};

template <typename ValueT, typename NodeT>
DEFINE_ADT_RC(DrrCtx, DrrCtxImpl<ValueT, NodeT>);

}  // namespace ap::drr

namespace ap::axpr {

template <typename ValueT, typename NodeT>
struct TypeImpl<drr::DrrCtx<ValueT, NodeT>> : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "DrrCtx"; }
};

}  // namespace ap::axpr
