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

#include "ap/axpr/method_class.h"
#include "ap/graph/node.h"
#include "ap/ir_match/ir_match_ctx.h"
#include "ap/ir_match/op_match_ctx.h"

namespace ap::ir_match {

template <typename ValueT, typename IrNodeT>
struct OpMatchCtxMethodClass {
  using This = OpMatchCtxMethodClass;
  using Self = ir_match::OpMatchCtx<IrNodeT>;

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, axpr::TryGetImpl<std::string>(attr_name_val));
    ADT_LET_CONST_REF(ir_op, GetIrOpByName(self, attr_name));
    if (ir_op.has_value()) {
      return ir_op.value();
    }
    return adt::errors::TypeError{
        std::string() + "'OpMatchCtx' has no attribute '" + attr_name + "'"};
  }

  using DrrValueT = drr::Value;
  using DrrNodeT = drr::Node<DrrValueT>;
  using DrrNativeIrOp = drr::NativeIrOp<DrrValueT, DrrNodeT>;
  using DrrPackedIrOp = drr::PackedIrOp<DrrValueT, DrrNodeT>;
  using PtnGraphNodeT = graph::Node<DrrNodeT>;

  using IrNativeIrOp = typename IrNodeT::native_op_type;
  using IrPackedIrOp = typename IrNodeT::packed_op_type;

  adt::Result<std::optional<ValueT>> GetIrOpByName(
      const Self& self, const std::string& attr_name) {
    ADT_LET_CONST_REF(ir_match_ctx, adt::WeakPtrLock(self->ir_mtach_ctx));
    const auto& source_pattern_ctx = ir_match_ctx->source_pattern_ctx;
    const auto& op_pattern_ctx = source_pattern_ctx->op_pattern_ctx;
    const auto& iter = op_pattern_ctx->uid2ir_op.find(attr_name);
    if (iter == op_pattern_ctx->uid2ir_op.end()) {
      return std::nullopt;
    }
    auto GetIrOpByPtnNode =
        [&](const PtnGraphNodeT& node) -> adt::Result<IrNodeT> {
      const auto& graph_match_ctx = ir_match_ctx->graph_match_ctx;
      return graph_match_ctx->GetSoleBigGraphNode(node);
    };
    ADT_LET_CONST_REF(
        ir_node,
        iter->second.Match(
            [&](const DrrNativeIrOp& native_ir_op) -> adt::Result<IrNodeT> {
              return GetIrOpByPtnNode(native_ir_op->node);
            },
            [&](const DrrPackedIrOp& packed_ir_op) -> adt::Result<IrNodeT> {
              return GetIrOpByPtnNode(packed_ir_op->node);
            },
            [&](const auto&) -> adt::Result<IrNodeT> {
              return adt::errors::ValueError{
                  std::string() + "Failed to get OpMatchCtx attribute, '" +
                  attr_name + "' is a unbounded op which should not be."};
            }));
    ADT_LET_CONST_REF(
        ir_op,
        ir_node.Match(
            [&](const IrNativeIrOp& impl) -> adt::Result<ValueT> {
              return ValueT{impl};
            },
            [&](const IrPackedIrOp& impl) -> adt::Result<ValueT> {
              return ValueT{impl};
            },
            [&](const auto&) -> adt::Result<ValueT> {
              return adt::errors::RuntimeError{
                  std::string() +
                  "a ptn op node has wrongly matched to a non-op ir node."};
            }));
    return ir_op;
  }
};

}  // namespace ap::ir_match

namespace ap::axpr {

template <typename ValueT, typename IrNodeT>
struct MethodClassImpl<ValueT, ir_match::OpMatchCtx<IrNodeT>>
    : public ir_match::OpMatchCtxMethodClass<ValueT, IrNodeT> {};

template <typename ValueT, typename IrNodeT>
struct MethodClassImpl<ValueT, TypeImpl<ir_match::OpMatchCtx<IrNodeT>>> {};

}  // namespace ap::axpr
