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
#include "ap/drr/unbound_packed_ir_op.h"
#include "ap/drr/unbound_packed_ir_value.h"
#include "ap/graph/tags.h"

namespace ap::drr {

template <typename ValueT, typename NodeT = Node<ValueT>>
using ValueImpl = ap::axpr::ValueBase<ValueT,
                                      UnboundIrValue<ValueT, NodeT>,
                                      UnboundPackedIrValue<ValueT, NodeT>,
                                      NativeIrOp<ValueT, NodeT>,
                                      PackedIrOp<ValueT, NodeT>,
                                      tSrcPtn<NativeIrOpDeclare<ValueT, NodeT>>,
                                      tSrcPtn<PackedIrOpDeclare<ValueT, NodeT>>,
                                      tSrcPtn<UnboundNativeIrOp<ValueT, NodeT>>,
                                      tSrcPtn<UnboundPackedIrOp<ValueT, NodeT>>,
                                      tSrcPtn<NativeIrValue<NodeT>>,
                                      tSrcPtn<PackedIrValue<NodeT>>,
                                      tSrcPtn<OpPatternCtx<ValueT, NodeT>>,
                                      tSrcPtn<TensorPatternCtx<ValueT, NodeT>>,
                                      tStarred<tSrcPtn<PackedIrValue<NodeT>>>,
                                      SourcePatternCtx<ValueT, NodeT>,
                                      tResPtn<NativeIrOpDeclare<ValueT, NodeT>>,
                                      tResPtn<PackedIrOpDeclare<ValueT, NodeT>>,
                                      tResPtn<UnboundNativeIrOp<ValueT, NodeT>>,
                                      tResPtn<UnboundPackedIrOp<ValueT, NodeT>>,
                                      tResPtn<NativeIrValue<NodeT>>,
                                      tResPtn<PackedIrValue<NodeT>>,
                                      tResPtn<OpPatternCtx<ValueT, NodeT>>,
                                      tResPtn<TensorPatternCtx<ValueT, NodeT>>,
                                      tStarred<tResPtn<PackedIrValue<NodeT>>>,
                                      ResultPatternCtx<ValueT, NodeT>,
                                      DrrCtx<ValueT, NodeT>>;

struct Value : public ValueImpl<Value> {
  using ValueImpl<Value>::ValueImpl;
  DEFINE_ADT_VARIANT_METHODS(ValueImpl<Value>);
};

using Val = Value;

using Env = ap::axpr::Environment<Val>;

using EnvMgr = ap::axpr::EnvironmentManager<Val>;

}  // namespace ap::drr
