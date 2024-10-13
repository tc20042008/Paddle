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
#include "ap/ir_match/ir_match_ctx.h"
#include "ap/ir_match/tensor_match_ctx.h"

namespace ap::ir_match {

template <typename ValueT, typename IrNodeT>
struct TensorMatchCtxMethodClass {
  using This = TensorMatchCtxMethodClass;
  using Self = ir_match::TensorMatchCtx<IrNodeT>;

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, axpr::TryGetImpl<std::string>(attr_name_val));
    ADT_LET_CONST_REF(ir_tensor, GetIrTensorByName(self, attr_name));
    if (ir_tensor.has_value()) {
      return ir_tensor.value();
    }
    return adt::errors::TypeError{std::string() +
                                  "'TensorMatchCtx' has no attribute '" +
                                  attr_name + "'"};
  }

  using DrrValueT = drr::Value;
  using DrrNodeT = drr::Node<DrrValueT>;
  using DrrNativeIrValue = drr::NativeIrValue<DrrNodeT>;
  using DrrPackedIrValue = drr::PackedIrValue<DrrNodeT>;
  using PtnGraphNodeT = graph::Node<DrrNodeT>;

  using IrNativeIrValue = typename IrNodeT::native_value_type;
  using IrPackedIrValue = typename IrNodeT::packed_value_type;

  adt::Result<std::optional<ValueT>> GetIrTensorByName(
      const Self& self, const std::string& attr_name) {
    ADT_LET_CONST_REF(ir_match_ctx, adt::WeakPtrLock(self->ir_mtach_ctx));
    const auto& source_pattern_ctx = ir_match_ctx->source_pattern_ctx;
    const auto& tensor_pattern_ctx = source_pattern_ctx->tensor_pattern_ctx;
    const auto& iter = tensor_pattern_ctx->uid2ir_value.find(attr_name);
    if (iter == tensor_pattern_ctx->uid2ir_value.end()) {
      return std::nullopt;
    }
    auto GetIrValueByPtnNode =
        [&](const PtnGraphNodeT& node) -> adt::Result<IrNodeT> {
      const auto& graph_match_ctx = ir_match_ctx->graph_match_ctx;
      return graph_match_ctx->GetSoleBigGraphNode(node);
    };
    ADT_LET_CONST_REF(
        ir_node,
        iter->second.Match(
            [&](const DrrNativeIrValue& native_ir_value)
                -> adt::Result<IrNodeT> {
              return GetIrValueByPtnNode(native_ir_value->node);
            },
            [&](const DrrPackedIrValue& packed_ir_value)
                -> adt::Result<IrNodeT> {
              return adt::errors::NotImplementedError{
                  "'TensorMatchCtx' has not supported packed tensors now."};
            },
            [&](const auto&) -> adt::Result<IrNodeT> {
              return adt::errors::ValueError{
                  std::string() + "Failed to get OpMatchCtx attribute, '" +
                  attr_name + "' is a unbounded op which should not be."};
            }));
    ADT_LET_CONST_REF(
        ir_value,
        ir_node.Match(
            [&](const IrNativeIrValue& impl) -> adt::Result<ValueT> {
              return ValueT{impl};
            },
            [&](const IrPackedIrValue& impl) -> adt::Result<ValueT> {
              return ValueT{impl};
            },
            [&](const auto&) -> adt::Result<ValueT> {
              return adt::errors::RuntimeError{
                  std::string() +
                  "a ptn op node has wrongly matched to a non-op ir node."};
            }));
    return ir_value;
  }
};

}  // namespace ap::ir_match

namespace ap::axpr {

template <typename ValueT, typename IrNodeT>
struct MethodClassImpl<ValueT, ir_match::TensorMatchCtx<IrNodeT>>
    : public ir_match::TensorMatchCtxMethodClass<ValueT, IrNodeT> {};

template <typename ValueT, typename IrNodeT>
struct MethodClassImpl<ValueT, TypeImpl<ir_match::TensorMatchCtx<IrNodeT>>> {};

}  // namespace ap::axpr
