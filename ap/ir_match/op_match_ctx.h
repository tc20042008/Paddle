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

namespace ap::ir_match {

template <typename IrNodeT>
struct IrMatchCtxImpl;

template <typename IrNodeT>
struct OpMatchCtxImpl {
  std::weak_ptr<IrMatchCtxImpl<IrNodeT>> ir_mtach_ctx;
};

template <typename IrNodeT>
DEFINE_ADT_RC(OpMatchCtx, OpMatchCtxImpl<IrNodeT>);

}  // namespace ap::ir_match

namespace ap::axpr {

template <typename IrNodeT>
struct TypeImpl<ir_match::OpMatchCtx<IrNodeT>> : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "OpMatchCtx"; }
};

}  // namespace ap::axpr
