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
#include "ap/axpr/type.h"

#include "ap/drr/ir_op.h"
#include "ap/drr/ir_value.h"
#include "ap/drr/op_pattern_ctx.h"
#include "ap/drr/tags.h"
#include "ap/drr/tensor_pattern_ctx.h"

namespace ap::drr {

template <typename ValueT, typename NodeT>
struct OpTensorPatternCtxHelper {
  using OpPtnCtx = OpPatternCtx<ValueT, NodeT>;
  using TensorPtnCtx = TensorPatternCtx<ValueT, NodeT>;

  adt::Result<ValueT> ConnectIrOpAndIrValue(
      const NativeIrOp<ValueT, NodeT>& native_ir_op,
      const adt::List<NativeIrValue<NodeT>>& inputs,
      const adt::List<NativeIrValue<NodeT>>& outputs) {
    ADT_LET_CONST_REF(op_upstream_nodes, native_ir_op->node.UpstreamNodes());
    ADT_CHECK(op_upstream_nodes.size() == 0);
    ADT_LET_CONST_REF(op_downstream_nodes,
                      native_ir_op->node.DownstreamNodes());
    ADT_CHECK(op_downstream_nodes.size() == 0);
    ADT_LET_CONST_REF(
        op_pattern_ctx,
        adt::WeakPtrLock(native_ir_op->op_declare->op_pattern_ctx));
    const auto& node_arena = op_pattern_ctx->node_arena;
    for (int i = 0; i < inputs->size(); ++i) {
      const auto& native_ir_op_operand = node_arena->New([&](const auto& node) {
        return NativeIrOpOperand<NodeT>{node, i};
      });
      ADT_RETURN_IF_ERR(
          inputs->at(i)->node.ConnectTo(native_ir_op_operand.node(),
                                        graph::UnindexedTag<std::monostate>{},
                                        graph::IndexedTag<std::monostate>{}));
      ADT_RETURN_IF_ERR(native_ir_op_operand.node().ConnectTo(
          native_ir_op->node,
          graph::IndexedTag<std::monostate>{},
          graph::IndexedTag<std::monostate>{}));
    }
    for (int i = 0; i < outputs->size(); ++i) {
      ADT_LET_CONST_REF(output_upstream_nodes,
                        outputs->at(i)->node.UpstreamNodes());
      ADT_CHECK(output_upstream_nodes.size() == 0);
      const auto& native_ir_op_result = node_arena->New([&](const auto& node) {
        return NativeIrOpResult<NodeT>{node, i};
      });
      ADT_RETURN_IF_ERR(
          native_ir_op->node.ConnectTo(native_ir_op_result.node(),
                                       graph::IndexedTag<std::monostate>{},
                                       graph::IndexedTag<std::monostate>{}));
      ADT_RETURN_IF_ERR(native_ir_op_result.node().ConnectTo(
          outputs->at(i)->node,
          graph::IndexedTag<std::monostate>{},
          graph::IndexedTag<std::monostate>{}));
    }
    SetIrOpByUid(op_pattern_ctx, native_ir_op->name, native_ir_op);
    return adt::Nothing{};
  }

  adt::Result<ValueT> ConnectIrOpAndIrValue(
      const PackedIrOp<ValueT, NodeT>& packed_ir_op,
      const adt::List<IrValue<NodeT>>& inputs,
      const adt::List<IrValue<NodeT>>& outputs) {
    ADT_LET_CONST_REF(op_upstream_nodes, packed_ir_op->node.UpstreamNodes());
    ADT_CHECK(op_upstream_nodes.size() == 0);
    ADT_LET_CONST_REF(op_downstream_nodes,
                      packed_ir_op->node.DownstreamNodes());
    ADT_CHECK(op_downstream_nodes.size() == 0);
    ADT_LET_CONST_REF(
        op_pattern_ctx,
        adt::WeakPtrLock(packed_ir_op->op_declare->op_pattern_ctx));
    const auto& node_arena = op_pattern_ctx->node_arena;
    for (int i = 0; i < inputs->size(); ++i) {
      const auto& packed_ir_op_operand = node_arena->New([&](const auto& node) {
        return PackedIrOpOperand<NodeT>{node, i};
      });
      ADT_RETURN_IF_ERR(
          inputs->at(i).node().ConnectTo(packed_ir_op_operand.node(),
                                         graph::UnindexedTag<std::monostate>{},
                                         graph::IndexedTag<std::monostate>{}));
      ADT_RETURN_IF_ERR(packed_ir_op_operand.node().ConnectTo(
          packed_ir_op->node,
          graph::IndexedTag<std::monostate>{},
          graph::UnindexedTag<std::monostate>{}));
    }
    for (int i = 0; i < outputs->size(); ++i) {
      ADT_LET_CONST_REF(output_upstream_nodes,
                        outputs->at(i).node().UpstreamNodes());
      ADT_CHECK(output_upstream_nodes.size() == 0);
      const auto& packed_ir_op_result = node_arena->New([&](const auto& node) {
        return PackedIrOpResult<NodeT>{node, i};
      });
      ADT_RETURN_IF_ERR(
          packed_ir_op->node.ConnectTo(packed_ir_op_result.node(),
                                       graph::UnindexedTag<std::monostate>{},
                                       graph::IndexedTag<std::monostate>{}));
      ADT_RETURN_IF_ERR(packed_ir_op_result.node().ConnectTo(
          outputs->at(i).node(),
          graph::IndexedTag<std::monostate>{},
          graph::IndexedTag<std::monostate>{}));
    }
    SetIrOpByUid(op_pattern_ctx, packed_ir_op->name, packed_ir_op);
    return adt::Nothing{};
  }

  adt::Result<IrOp<ValueT, NodeT>> GetIrOpByUid(const OpPtnCtx& self,
                                                const std::string& name) {
    const auto& iter = self->uid2ir_op.find(name);
    if (iter == self->uid2ir_op.end()) {
      return adt::errors::AttributeError{std::string() + "no op named '" +
                                         name + "' registered."};
    }
    return iter->second;
  }

  template <typename OpPtnCtxT>
  bool HasIrOpByUid(const OpPtnCtxT& self, const std::string& name) {
    return self->uid2ir_op.count(name) > 0;
  }

  template <typename OpPtnCtxT>
  void SetIrOpByUid(const OpPtnCtxT& self,
                    const std::string& name,
                    const IrOp<ValueT, NodeT>& ir_op) {
    self->uid2ir_op[name] = ir_op;
  }

  template <typename TensorPtnCtxT>
  bool HasIrValueByUid(const TensorPtnCtxT& self, const std::string& name) {
    return self->uid2ir_value.count(name);
  }

  template <typename TensorPtnCtxT>
  adt::Result<IrValue<NodeT>> GetIrValueByUid(const TensorPtnCtxT& self,
                                              const std::string& name) {
    const auto& iter = self->uid2ir_value.find(name);
    if (iter == self->uid2ir_value.end()) {
      return adt::errors::AttributeError{std::string() + "no tensor named '" +
                                         name + "' registered."};
    }
    return iter->second;
  }

  template <typename TensorPtnCtxT>
  void SetIrValueByUid(const TensorPtnCtxT& self,
                       const std::string& name,
                       const IrValue<NodeT>& ir_value) {
    self->uid2ir_value[name] = ir_value;
  }

  adt::Result<NativeIrValue<NodeT>> CloneIrValueDataAndRegister(
      const TensorPtnCtx& self, const NativeIrValue<NodeT>& native_ir_value) {
    const auto& cloned_node = self->node_arena->New([&](const auto& node) {
      return NativeIrValue<NodeT>{node, native_ir_value->name};
    });
    ADT_CHECK(cloned_node.template Has<NativeIrValue<NodeT>>());
    const auto& cloned = cloned_node.template Get<NativeIrValue<NodeT>>();
    SetIrValueByUid(self, native_ir_value->name, cloned);
    return cloned;
  }

  adt::Result<PackedIrValue<NodeT>> CloneIrValueDataAndRegister(
      const TensorPtnCtx& self, const PackedIrValue<NodeT>& packed_ir_value) {
    const auto& cloned_node = self->node_arena->New([&](const auto& node) {
      return PackedIrValue<NodeT>{node, packed_ir_value->name};
    });
    ADT_CHECK(cloned_node.template Has<PackedIrValue<NodeT>>());
    const auto& cloned = cloned_node.template Get<PackedIrValue<NodeT>>();
    SetIrValueByUid(self, packed_ir_value->name, cloned);
    return cloned;
  }

  adt::Result<NativeIrOp<ValueT, NodeT>> GetNativeIrOpByUnboundNativeIrOp(
      const UnboundNativeIrOp<ValueT, NodeT>& ir_op) {
    ADT_LET_CONST_REF(op_pattern_ctx,
                      adt::WeakPtrLock(ir_op->op_declare->op_pattern_ctx));
    const auto& node = op_pattern_ctx->node_arena->New([&](const auto& node) {
      return NativeIrOp<ValueT, NodeT>{node, ir_op->op_declare, ir_op->name};
    });
    ADT_CHECK(node.template Has<NativeIrOp<ValueT, NodeT>>());
    return node.template Get<NativeIrOp<ValueT, NodeT>>();
  }

  adt::Result<PackedIrOp<ValueT, NodeT>> GetPackedIrOpByUnboundPackedIrOp(
      const UnboundPackedIrOp<ValueT, NodeT>& ir_op) {
    ADT_LET_CONST_REF(op_pattern_ctx,
                      adt::WeakPtrLock(ir_op->op_declare->op_pattern_ctx));
    const auto& node = op_pattern_ctx->node_arena->New([&](const auto& node) {
      return PackedIrOp<ValueT, NodeT>{node, ir_op->op_declare, ir_op->name};
    });
    ADT_CHECK(node.template Has<PackedIrOp<ValueT, NodeT>>());
    return node.template Get<PackedIrOp<ValueT, NodeT>>();
  }

  adt::Result<NativeIrValue<NodeT>> GetNativeIrValueByUnboundIrValue(
      const UnboundIrValue<ValueT, NodeT>& ir_value) {
    ADT_LET_CONST_REF(tensor_ctx,
                      adt::WeakPtrLock(ir_value->tensor_pattern_ctx));
    if (HasIrValueByUid(tensor_ctx, ir_value->name)) {
      ADT_LET_CONST_REF(ir_value, GetIrValueByUid(tensor_ctx, ir_value->name));
      const auto& opt_ret = ir_value.Match(
          [](const NativeIrValue<NodeT>& impl)
              -> adt::Result<NativeIrValue<NodeT>> { return impl; },
          [&](const auto&) -> adt::Result<NativeIrValue<NodeT>> {
            return adt::errors::RuntimeError{"only NativeIrValue supported."};
          });
      ADT_LET_CONST_REF(ret, opt_ret);
      return ret;
    }
    const auto& node_arena = tensor_ctx->node_arena;
    const auto& node = node_arena->New([&](const auto& node) {
      return NativeIrValue<NodeT>{node, ir_value->name};
    });
    ADT_CHECK(node.template Has<NativeIrValue<NodeT>>());
    const auto& native_ir_value = node.template Get<NativeIrValue<NodeT>>();
    SetIrValueByUid(tensor_ctx, native_ir_value->name, native_ir_value);
    return native_ir_value;
  }

  adt::Result<PackedIrValue<NodeT>> GetPackedIrValueByUnboundPackedIrValue(
      const UnboundPackedIrValue<ValueT, NodeT>& ir_value) {
    ADT_LET_CONST_REF(tensor_ctx,
                      adt::WeakPtrLock(ir_value->tensor_pattern_ctx));
    if (HasIrValueByUid(tensor_ctx, ir_value->name)) {
      ADT_LET_CONST_REF(ir_value, GetIrValueByUid(tensor_ctx, ir_value->name));
      const auto& opt_ret = ir_value.Match(
          [](const PackedIrValue<NodeT>& impl)
              -> adt::Result<PackedIrValue<NodeT>> { return impl; },
          [&](const auto&) -> adt::Result<PackedIrValue<NodeT>> {
            return adt::errors::RuntimeError{"only PackedIrValue supported."};
          });
      ADT_LET_CONST_REF(ret, opt_ret);
      return ret;
    }
    const auto& node_arena = tensor_ctx->node_arena;
    const auto& node = node_arena->New([&](const auto& node) {
      return PackedIrValue<NodeT>{node, ir_value->name};
    });
    ADT_CHECK(node.template Has<PackedIrValue<NodeT>>());
    const auto& packed_ir_value = node.template Get<PackedIrValue<NodeT>>();
    SetIrValueByUid(tensor_ctx, packed_ir_value->name, packed_ir_value);
    return packed_ir_value;
  }
};

}  // namespace ap::drr
