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
#include "ap/include/adt/adt.h"
#include "ap/include/axpr/value.h"
#include "ap/include/drr/op_pattern_ctx.h"
#include "ap/include/drr/tags.h"
#include "ap/include/drr/tensor_pattern_ctx.h"
#include "ap/include/drr/type.h"
#include "ap/include/graph/node_arena.h"

namespace ap::drr {

struct SourcePatternCtxImpl {
  std::shared_ptr<graph::NodeArena<drr::Node>> node_arena;
  OpPatternCtx op_pattern_ctx;
  TensorPatternCtx tensor_pattern_ctx;

  bool operator==(const SourcePatternCtxImpl& other) const {
    return this != &other;
  }
};

DEFINE_ADT_RC(SourcePatternCtx, SourcePatternCtxImpl);

const axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>&
GetSourcePatternCtxClass();

template <>
struct Type<drr::SourcePatternCtx> : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "SourcePatternCtx"; }

  static const axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>&
  GetClass() {
    return GetSourcePatternCtxClass();
  }
};

}  // namespace ap::drr
