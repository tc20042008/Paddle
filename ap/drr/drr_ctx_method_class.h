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

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, axpr::TryGetImpl<std::string>(attr_name_val));
    if (attr_name == "source_pattern") {
      return axpr::Method<ValueT>{self, &This::InitSourcePattern};
    } else if (attr_name == "result_pattern") {
      return axpr::Method<ValueT>{self, &This::InitResultPattern};
    } else if (attr_name == "demo_graph") {
      return axpr::Method<ValueT>{self, &This::InitDemoGraph};
    } else if (attr_name == "test_source_pattern_by_demo_graph") {
      return axpr::Method<ValueT>{self, &This::TestSourcePatternByDemoGraph};
    } else {
      return adt::errors::AttributeError{std::string() +
                                         "'DrrCtx' object has no attribute '" +
                                         attr_name + "'"};
    }
  }

  adt::Result<ValueT> SetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, axpr::TryGetImpl<std::string>(attr_name_val));
    if (attr_name == "pass_name") {
      return axpr::Method<ValueT>{self, &This::SetPassName};
    } else {
      return adt::errors::AttributeError{std::string() +
                                         "'DrrCtx' object has no attribute '" +
                                         attr_name + "'"};
    }
  }

  static adt::Result<ValueT> SetPassName(const ValueT& self_val,
                                         const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, axpr::TryGetImpl<Self>(self_val));
    ADT_CHECK(args.size() == 2);
    ADT_LET_CONST_REF(pass_name, axpr::TryGetImpl<std::string>(args.at(1)))
        << adt::errors::TypeError{
               std::string() +
               "DrrCtx.pass_name should be set to a 'str'. but '" +
               axpr::GetTypeName(args.at(0)) + "' were given."};
    self.shared_ptr()->pass_name = pass_name;
    return adt::Nothing{};
  }

  static adt::Result<ValueT> InitSourcePattern(
      const axpr::ApplyT<ValueT>& Apply,
      const ValueT& self_val,
      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, axpr::TryGetImpl<Self>(self_val));
    if (self->source_pattern_ctx.has_value()) {
      return adt::errors::ValueError{
          "'DrrCtx.source_pattern' has been initialized."};
    }
    if (args.size() != 1) {
      return adt::errors::TypeError{
          std::string() + "DrrCtx.source_pattern takes 1 argument. but " +
          std::to_string(args.size()) + " were given."};
    }
    auto node_arena = std::make_shared<graph::NodeArena<NodeT>>();
    SourcePatternCtx<ValueT, NodeT> source_pattern_ctx{
        node_arena,
        OpPatternCtx<ValueT, NodeT>{
            node_arena,
            std::map<std::string, IrOp<ValueT, NodeT>>{},
        },
        TensorPatternCtx<ValueT, NodeT>{node_arena,
                                        std::map<std::string, IrValue<NodeT>>{},
                                        self.shared_ptr()}};
    self.shared_ptr()->source_pattern_ctx = source_pattern_ctx;
    ADT_RETURN_IF_ERR(Apply(args.at(0),
                            {SrcPtn(source_pattern_ctx->op_pattern_ctx),
                             SrcPtn(source_pattern_ctx->tensor_pattern_ctx)}));
    return adt::Nothing{};
  }

  static adt::Result<ValueT> InitDemoGraph(const axpr::ApplyT<ValueT>& Apply,
                                           const ValueT& self_val,
                                           const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, axpr::TryGetImpl<Self>(self_val));
    if (self->demo_graph_source_pattern_ctx.has_value()) {
      return adt::errors::ValueError{
          "'DrrCtx.demo_graph' has been initialized."};
    }
    if (args.size() != 1) {
      return adt::errors::TypeError{
          std::string() + "DrrCtx.demo_graph takes 1 argument. but " +
          std::to_string(args.size()) + " were given."};
    }
    auto node_arena = std::make_shared<graph::NodeArena<NodeT>>();
    SourcePatternCtx<ValueT, NodeT> demo_graph_source_pattern_ctx{
        node_arena,
        OpPatternCtx<ValueT, NodeT>{
            node_arena,
            std::map<std::string, IrOp<ValueT, NodeT>>{},
        },
        TensorPatternCtx<ValueT, NodeT>{node_arena,
                                        std::map<std::string, IrValue<NodeT>>{},
                                        self.shared_ptr()}};
    self.shared_ptr()->demo_graph_source_pattern_ctx =
        demo_graph_source_pattern_ctx;
    ADT_RETURN_IF_ERR(
        Apply(args.at(0),
              {SrcPtn(demo_graph_source_pattern_ctx->op_pattern_ctx),
               SrcPtn(demo_graph_source_pattern_ctx->tensor_pattern_ctx)}));
    return adt::Nothing{};
  }

  static adt::Result<ValueT> TestSourcePatternByDemoGraph(
      const ValueT& self_val, const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, axpr::TryGetImpl<Self>(self_val));
    ADT_CHECK(self->demo_graph_source_pattern_ctx.has_value());
    ADT_CHECK(self->source_pattern_ctx.has_value());
    const auto& obj_node_arena =
        self->demo_graph_source_pattern_ctx.value()->node_arena;
    const auto& ptn_node_arena = self->source_pattern_ctx.value()->node_arena;
    ADT_LET_CONST_REF(is_matched,
                      DemoGraphHelper<ValueT>{}.IsGraphMatched(
                          *obj_node_arena, *ptn_node_arena));
    return is_matched;
  }

  static adt::Result<ValueT> InitResultPattern(
      const axpr::ApplyT<ValueT>& Apply,
      const ValueT& self_val,
      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, axpr::TryGetImpl<Self>(self_val));
    if (!self->source_pattern_ctx.has_value()) {
      return adt::errors::ValueError{
          "'DrrCtx.source_pattern' has not been initialized."};
    }
    if (self->result_pattern_ctx.has_value()) {
      return adt::errors::ValueError{
          "'DrrCtx.result_pattern' has been initialized."};
    }
    if (args.size() != 1) {
      return adt::errors::TypeError{
          std::string() + "DrrCtx.result_pattern takes 1 argument. but " +
          std::to_string(args.size()) + " were given."};
    }
    auto node_arena = std::make_shared<graph::NodeArena<NodeT>>();
    ResultPatternCtx<ValueT, NodeT> result_pattern_ctx{
        node_arena,
        OpPatternCtx<ValueT, NodeT>{
            node_arena,
            std::map<std::string, IrOp<ValueT, NodeT>>{},
        },
        TensorPatternCtx<ValueT, NodeT>{node_arena,
                                        std::map<std::string, IrValue<NodeT>>{},
                                        self.shared_ptr()},
        self->source_pattern_ctx.value()};
    self.shared_ptr()->result_pattern_ctx = result_pattern_ctx;
    ADT_RETURN_IF_ERR(Apply(args.at(0),
                            {ResPtn(result_pattern_ctx->op_pattern_ctx),
                             ResPtn(result_pattern_ctx->tensor_pattern_ctx)}));
    return adt::Nothing{};
  }
};

}  // namespace ap::drr

namespace ap::axpr {

template <typename ValueT, typename NodeT>
struct MethodClassImpl<ValueT, drr::DrrCtx<ValueT, NodeT>>
    : public drr::DrrCtxMethodClass<ValueT, NodeT> {};

template <typename ValueT, typename NodeT>
struct MethodClassImpl<ValueT, TypeImpl<drr::DrrCtx<ValueT, NodeT>>> {};

}  // namespace ap::axpr
