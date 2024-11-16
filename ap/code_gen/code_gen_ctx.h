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
#include "ap/axpr/builtin_object.h"
#include "ap/axpr/core_expr.h"
#include "ap/axpr/type.h"
#include "ap/code_gen/arg_source_ctx.h"
#include "ap/code_module/adt.h"
#include "ap/code_module/arg_type.h"
#include "ap/code_module/data_type.h"
#include "ap/drr/value.h"
#include "ap/ir_match/ir_match_ctx.h"
#include "paddle/cinn/adt/adt.h"

namespace ap::code_gen {

template <typename BirNode>
struct CodeGenCtxImpl {
  std::optional<ir_match::IrMatchCtx<BirNode>> ir_match_ctx;

  using DrrValue = drr::Value;
  using DrrNode = drr::Node<DrrValue>;
  using DrrPackedIrOp = drr::PackedIrOp<DrrValue, DrrNode>;

  DrrPackedIrOp res_ptn_ir_op;

  ArgSourceCtx<BirNode> arg_source_ctx;

  bool operator==(const CodeGenCtxImpl& other) const { return this == &other; }
};

template <typename BirNode>
DEFINE_ADT_RC(CodeGenCtx, CodeGenCtxImpl<BirNode>);

}  // namespace ap::code_gen

namespace ap::axpr {

template <typename BirNode>
struct TypeImpl<ap::code_gen::CodeGenCtx<BirNode>> : public std::monostate {
  using value_type = ap::code_gen::CodeGenCtx<BirNode>;

  const char* Name() const { return "CodeGenCtx"; }
};

}  // namespace ap::axpr
