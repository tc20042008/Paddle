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
#include "ap/drr/demo_graph_helper.h"
#include "ap/drr/drr_ctx.h"
#include "ap/drr/ir_op.h"
#include "ap/drr/ir_value.h"
#include "ap/drr/op_pattern_ctx.h"
#include "ap/drr/tags.h"
#include "ap/drr/tensor_pattern_ctx.h"

namespace ap::drr {

template <typename ValueT, typename NodeT>
struct DrrCtxMethodClass {
  using This = DrrCtxMethodClass;
  using Self = drr::DrrCtx<ValueT, NodeT>;
};

template <typename ValueT, typename NodeT>
struct TypeImplDrrCtxMethodClass {
  using This = TypeImplDrrCtxMethodClass;
  using Self = axpr::TypeImpl<DrrCtx<ValueT, NodeT>>;

  adt::Result<ValueT> Call(const Self&) { return &This::StaticConstruct; }

  static adt::Result<ValueT> StaticConstruct(const axpr::ApplyT<ValueT>& Apply,
                                             const ValueT&,
                                             const std::vector<ValueT>& args) {
    return This{}.Construct(Apply, args);
  }

  adt::Result<ValueT> Construct(const axpr::ApplyT<ValueT>& Apply,
                                const std::vector<ValueT>& packed_args_val) {
    DrrCtx<ValueT, NodeT> self{};
    const auto& packed_args = axpr::CastToPackedArgs(packed_args_val);
    const auto& [args, kwargs] = *packed_args;
    ADT_CHECK(args->size() == 0) << adt::errors::TypeError{
        "the constructor of DrrCtx takes keyword arguments only."};
    {
      ADT_LET_CONST_REF(
          def_source_pattern,
          kwargs->template Get<axpr::Closure<ValueT>>("source_pattern"));
      auto node_arena = std::make_shared<graph::NodeArena<NodeT>>();
      SourcePatternCtx<ValueT, NodeT> source_pattern_ctx{
          node_arena,
          OpPatternCtx<ValueT, NodeT>{
              node_arena,
              std::map<std::string, IrOp<ValueT, NodeT>>{},
              self.shared_ptr()},
          TensorPatternCtx<ValueT, NodeT>{
              node_arena,
              std::map<std::string, IrValue<NodeT>>{},
              self.shared_ptr()}};
      self.shared_ptr()->source_pattern_ctx = source_pattern_ctx;
      ADT_RETURN_IF_ERR(
          Apply(def_source_pattern,
                {SrcPtn(source_pattern_ctx->op_pattern_ctx),
                 SrcPtn(source_pattern_ctx->tensor_pattern_ctx)}));
    }
    {
      ADT_LET_CONST_REF(
          def_result_pattern,
          kwargs->template Get<axpr::Closure<ValueT>>("result_pattern"));

      auto node_arena = std::make_shared<graph::NodeArena<NodeT>>();
      ResultPatternCtx<ValueT, NodeT> result_pattern_ctx{
          node_arena,
          OpPatternCtx<ValueT, NodeT>{
              node_arena,
              std::map<std::string, IrOp<ValueT, NodeT>>{},
              self.shared_ptr()},
          TensorPatternCtx<ValueT, NodeT>{
              node_arena,
              std::map<std::string, IrValue<NodeT>>{},
              self.shared_ptr()},
          self->source_pattern_ctx.value()};
      self.shared_ptr()->result_pattern_ctx = result_pattern_ctx;
      ADT_RETURN_IF_ERR(
          Apply(def_result_pattern,
                {ResPtn(result_pattern_ctx->op_pattern_ctx),
                 ResPtn(result_pattern_ctx->tensor_pattern_ctx)}));
    }
    return self;
  }
};

}  // namespace ap::drr

namespace ap::axpr {

template <typename ValueT, typename NodeT>
struct MethodClassImpl<ValueT, drr::DrrCtx<ValueT, NodeT>>
    : public drr::DrrCtxMethodClass<ValueT, NodeT> {};

template <typename ValueT, typename NodeT>
struct MethodClassImpl<ValueT, TypeImpl<drr::DrrCtx<ValueT, NodeT>>>
    : public drr::TypeImplDrrCtxMethodClass<ValueT, NodeT> {};

}  // namespace ap::axpr
