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
#include <typeindex>
#include "ap/axpr/adt.h"
#include "ap/axpr/value.h"
#include "ap/drr/drr_ctx.h"
#include "ap/drr/native_ir_op.h"
#include "ap/drr/native_ir_value.h"
#include "ap/drr/node.h"
#include "ap/drr/op_pattern_ctx.h"
#include "ap/drr/packed_ir_op.h"
#include "ap/drr/packed_ir_value.h"
#include "ap/drr/result_pattern_ctx.h"
#include "ap/drr/source_pattern_ctx.h"
#include "ap/drr/tags.h"
#include "ap/drr/tensor_pattern_ctx.h"
#include "ap/drr/unbound_ir_value.h"
#include "ap/drr/unbound_native_ir_op.h"
#include "ap/drr/unbound_opt_packed_ir_op.h"
#include "ap/drr/unbound_packed_ir_op.h"
#include "ap/drr/unbound_packed_ir_value.h"
#include "ap/graph/tags.h"

namespace ap::drr {

using DrrValueImpl = std::variant<axpr::Value,
                                  UnboundIrValue<drr::Node>,
                                  UnboundPackedIrValue<drr::Node>,
                                  NativeIrOp<drr::Node>,
                                  PackedIrOp<drr::Node>,
                                  OptPackedIrOp<drr::Node>,
                                  tSrcPtn<NativeIrOpDeclare<drr::Node>>,
                                  tSrcPtn<PackedIrOpDeclare<drr::Node>>,
                                  OptPackedIrOpDeclare<drr::Node>,
                                  tSrcPtn<UnboundNativeIrOp<drr::Node>>,
                                  tSrcPtn<UnboundPackedIrOp<drr::Node>>,
                                  UnboundOptPackedIrOp<drr::Node>,
                                  tSrcPtn<NativeIrValue<drr::Node>>,
                                  tSrcPtn<PackedIrValue<drr::Node>>,
                                  tSrcPtn<OpPatternCtx>,
                                  tSrcPtn<TensorPatternCtx>,
                                  tStarred<tSrcPtn<PackedIrValue<drr::Node>>>,
                                  SourcePatternCtx,
                                  tResPtn<NativeIrOpDeclare<drr::Node>>,
                                  tResPtn<PackedIrOpDeclare<drr::Node>>,
                                  tResPtn<UnboundNativeIrOp<drr::Node>>,
                                  tResPtn<UnboundPackedIrOp<drr::Node>>,
                                  tResPtn<NativeIrValue<drr::Node>>,
                                  tResPtn<PackedIrValue<drr::Node>>,
                                  tResPtn<OpPatternCtx>,
                                  tResPtn<TensorPatternCtx>,
                                  tStarred<tResPtn<PackedIrValue<drr::Node>>>,
                                  ResultPatternCtx,
                                  DrrCtx>;

struct DrrValue : public DrrValueImpl {
  using DrrValueImpl::DrrValueImpl;
  DEFINE_ADT_VARIANT_METHODS(DrrValueImpl);

  template <typename... Args>
  decltype(auto) DrrValueMatch(Args&&... args) const {
    return Match(std::forward<Args>(args)...);
  }
};

}  // namespace ap::drr
