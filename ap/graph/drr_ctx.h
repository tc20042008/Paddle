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
#include "ap/graph/tags.h"

namespace ap::drr {

template <typename ValueT, typename NodeT>
struct DrrCtxImpl {
  SourcePatternCtx<ValueT, NodeT> source_pattern_ctx;
  ResultPatternCtx<ValueT, NodeT> result_pattern_ctx;
};

template <typename ValueT, typename NodeT>
DEFINE_ADT_RC(DrrCtx, DrrCtxImpl<ValueT, NodeT>);

}  // namespace ap::drr
