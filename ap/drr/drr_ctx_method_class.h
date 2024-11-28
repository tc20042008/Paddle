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

#include "ap/axpr/builtin_high_order_func_type.h"
#include "ap/axpr/method_class.h"
#include "ap/axpr/type.h"
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

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    if (attr_name == "init_pass_name") {
      return axpr::Method<ValueT>{self, &This::StaticInitPassName};
    }
    if (attr_name == "init_source_pattern") {
      return axpr::Method<ValueT>{self, &This::StaticInitSourcePattern};
    }
    if (attr_name == "init_constraint_func") {
      return axpr::Method<ValueT>{self, &This::StaticInitConstraintFunc};
    }
    if (attr_name == "init_result_pattern") {
      return axpr::Method<ValueT>{self, &This::StaticInitResultPattern};
    }
    return adt::errors::TypeError{
        std::string() + "DrrCtx object has no attribute '" + attr_name + "'"};
  }

  static adt::Result<ValueT> StaticInitPassName(
      axpr::InterpreterBase<ValueT>* interpreter,
      const ValueT& self_val,
      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template TryGet<Self>());
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(pass_name, args.at(0).template TryGet<std::string>())
        << adt::errors::TypeError{
               std::string() +
               "DrrCtx.init_pass_name() missing str typed argument 1"};
    self.shared_ptr()->pass_name = pass_name;
    return adt::Nothing{};
  }

  static adt::Result<ValueT> StaticInitSourcePattern(
      axpr::InterpreterBase<ValueT>* interpreter,
      const ValueT& self_val,
      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template TryGet<Self>());
    ADT_CHECK(!self->source_pattern_ctx.has_value());
    ADT_CHECK(args.size() == 1);
    const auto& def_source_pattern = args.at(0);
    auto node_arena = std::make_shared<graph::NodeArena<NodeT>>();
    SourcePatternCtx<ValueT, NodeT> source_pattern_ctx{
        node_arena,
        OpPatternCtx<ValueT, NodeT>{
            node_arena,
            std::map<std::string, IrOp<ValueT, NodeT>>{},
            self.shared_ptr()},
        TensorPatternCtx<ValueT, NodeT>{node_arena,
                                        std::map<std::string, IrValue<NodeT>>{},
                                        self.shared_ptr()}};
    self.shared_ptr()->source_pattern_ctx = source_pattern_ctx;
    ADT_RETURN_IF_ERR(interpreter->InterpretCall(
        def_source_pattern,
        {SrcPtn(source_pattern_ctx->op_pattern_ctx),
         SrcPtn(source_pattern_ctx->tensor_pattern_ctx)}));
    return adt::Nothing{};
  }

  static adt::Result<ValueT> StaticInitConstraintFunc(
      axpr::InterpreterBase<ValueT>* interpreter,
      const ValueT& self_val,
      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template TryGet<Self>());
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(
        constraint_func,
        args.at(0).template TryGet<axpr::Function<axpr::SerializableValue>>())
        << adt::errors::TypeError{std::string() +
                                  "DrrCtx.init_constraint_func() missing "
                                  "function typed argument 1"};
    self.shared_ptr()->constraint_func = constraint_func;
    return adt::Nothing{};
  }

  static adt::Result<ValueT> StaticInitResultPattern(
      axpr::InterpreterBase<ValueT>* interpreter,
      const ValueT& self_val,
      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template TryGet<Self>());
    ADT_CHECK(!self->result_pattern_ctx.has_value());
    ADT_CHECK(args.size() == 1);
    const auto& def_result_pattern = args.at(0);
    auto node_arena = std::make_shared<graph::NodeArena<NodeT>>();
    ResultPatternCtx<ValueT, NodeT> result_pattern_ctx{
        node_arena,
        OpPatternCtx<ValueT, NodeT>{
            node_arena,
            std::map<std::string, IrOp<ValueT, NodeT>>{},
            self.shared_ptr()},
        TensorPatternCtx<ValueT, NodeT>{node_arena,
                                        std::map<std::string, IrValue<NodeT>>{},
                                        self.shared_ptr()},
        self->source_pattern_ctx.value()};
    self.shared_ptr()->result_pattern_ctx = result_pattern_ctx;
    ADT_RETURN_IF_ERR(interpreter->InterpretCall(
        def_result_pattern,
        {ResPtn(result_pattern_ctx->op_pattern_ctx),
         ResPtn(result_pattern_ctx->tensor_pattern_ctx)}));
    return adt::Nothing{};
  }
};

template <typename ValueT, typename NodeT>
struct TypeImplDrrCtxMethodClass {
  using This = TypeImplDrrCtxMethodClass;
  using Self = axpr::TypeImpl<DrrCtx<ValueT, NodeT>>;

  adt::Result<ValueT> Call(const Self&) {
    return ValueT{&This::StaticConstruct};
  }

  static adt::Result<ValueT> StaticConstruct(
      axpr::InterpreterBase<ValueT>* interpreter,
      const ValueT&,
      const std::vector<ValueT>& args) {
    return This{}.Construct(interpreter, args);
  }

  adt::Result<ValueT> Construct(axpr::InterpreterBase<ValueT>* interpreter,
                                const std::vector<ValueT>& packed_args_val) {
    DrrCtx<ValueT, NodeT> self{};
    if (packed_args_val.size() == 0) {
      return self;
    }
    const auto& packed_args = axpr::CastToPackedArgs(packed_args_val);
    const auto& [args, kwargs] = *packed_args;
    ADT_CHECK(args->size() == 0) << adt::errors::TypeError{
        "the constructor of DrrCtx takes keyword arguments only."};
    {
      ADT_LET_CONST_REF(def_source_pattern, kwargs->Get("source_pattern"));
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
      ADT_RETURN_IF_ERR(interpreter->InterpretCall(
          def_source_pattern,
          {SrcPtn(source_pattern_ctx->op_pattern_ctx),
           SrcPtn(source_pattern_ctx->tensor_pattern_ctx)}));
    }
    {
      ADT_LET_CONST_REF(def_result_pattern, kwargs->Get("result_pattern"));
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
      ADT_RETURN_IF_ERR(interpreter->InterpretCall(
          def_result_pattern,
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
