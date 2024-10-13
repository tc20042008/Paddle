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
#include "ap/axpr/core_expr.h"
#include "ap/axpr/object.h"
#include "ap/axpr/type.h"
#include "ap/ir_match/ir_match_ctx.h"
#include "ap/kernel_define/adt.h"
#include "ap/kernel_define/arg_type.h"
#include "ap/kernel_define/data_type.h"
#include "ap/kernel_define/kernel_arg.h"
#include "paddle/cinn/adt/adt.h"

namespace ap::kernel_define {

struct NamedKernelArg {
  std::string arg_name;
  KernelArg kernel_arg;
};

template <typename IrNodeT>
struct DefineCtxImpl {
  std::optional<ir_match::IrMatchCtx<IrNodeT>> ir_match_ctx;

  std::vector<NamedKernelArg> registered_named_kernel_args;

  bool operator==(const DefineCtxImpl& other) const { return this == &other; }
};

template <typename IrNodeT>
DEFINE_ADT_RC(DefineCtx, DefineCtxImpl<IrNodeT>);

}  // namespace ap::kernel_define

namespace ap::axpr {

template <typename IrNodeT>
struct TypeImpl<ap::kernel_define::DefineCtx<IrNodeT>> : public std::monostate {
  using value_type = ap::kernel_define::DefineCtx<IrNodeT>;

  const char* Name() const { return "DefineCtx"; }
};

}  // namespace ap::axpr
