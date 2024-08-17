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

#include "paddle/common/enforce.h"
#include "paddle/phi/core/pexpr/index_lambda_constants.h"
#include "paddle/phi/core/pexpr/lambda_expr_builder.h"

namespace pexpr {

class IndexLambdaBuilder;

class IndexLambdaBuildContext {
 public:
  IndexLambdaBuildContext(const IndexLambdaBuildContext&) = delete;
  IndexLambdaBuildContext(IndexLambdaBuildContext&&) = delete;

  using var_type = LetVar;

  LetVar& Var(const std::string& name) { return let_ctx_->Var(name); }

  AnfExpr Bool(bool c) { return let_ctx_->Bool(c); }

  AnfExpr Int64(int64_t c) { return let_ctx_->Int64(c); }

  AnfExpr String(const std::string& c) { return let_ctx_->String(c); }

  AnfExpr PrimitiveOp(const std::string& c) {
    return let_ctx_->PrimitiveOp(pexpr::PrimitiveOp{c});
  }

  AnfExpr Lambda(const std::vector<std::string>& args,
                 const AnfExpr& anf_expr) {
    return let_ctx_->Lambda(MakeLambdaArgs(args), anf_expr);
  }

  template <typename... Args>
  AnfExpr MakeTensorIndexes(Args&&... args) {
    return let_ctx_->Call(kMakeTensorIndexes, std::forward<Args>(args)...);
  }

  template <typename... Args>
  AnfExpr InputTensorIndexes(Args&&... args) {
    return let_ctx_->Call(kInputTensorIndexes, std::forward<Args>(args)...);
  }

  template <typename... Args>
  AnfExpr OutputTensorIndexes(Args&&... args) {
    return let_ctx_->Call(kOutputTensorIndexes, std::forward<Args>(args)...);
  }

  template <typename... Args>
  AnfExpr IndexList(Args&&... args) {
    return let_ctx_->Call(kIndexList, std::forward<Args>(args)...);
  }

  template <typename... Args>
  AnfExpr BroadcastIndexMask(Args&&... args) {
    return let_ctx_->Call(kBroadcastIndexMask, std::forward<Args>(args)...);
  }

  template <typename... Args>
  AnfExpr IndexDot(Args&&... args) {
    return let_ctx_->Call(kIndexDot, std::forward<Args>(args)...);
  }

  template <typename... Args>
  AnfExpr IndexUnDot(Args&&... args) {
    return let_ctx_->Call(kIndexUnDot, std::forward<Args>(args)...);
  }

  AnfExpr IntArrayLikeIndexes() {
    return AnfExprBuilder().Var(kIntArrayLikeIndexes);
  }

  AnfExpr Undefined() { return AnfExprBuilder().Var(kUndefined); }

  AnfExpr Nothing() { return AnfExprBuilder().Var(kNothing); }

 private:
  friend class IndexLambdaBuilder;

  explicit IndexLambdaBuildContext(LetContext* let_ctx) : let_ctx_(let_ctx) {}

  std::vector<tVar<std::string>> MakeLambdaArgs(
      const std::vector<std::string>& args) {
    std::vector<tVar<std::string>> lambda_args;
    lambda_args.reserve(args.size());
    for (const auto& arg : args) {
      lambda_args.emplace_back(arg);
    }
    return lambda_args;
  }

  LetContext* let_ctx_;
};

class IndexLambdaBuilder {
 public:
  IndexLambdaBuilder() : lmbd_builder_() {}
  explicit IndexLambdaBuilder(const std::function<size_t()>& SeqNoGenerator)
      : lmbd_builder_(SeqNoGenerator) {}
  IndexLambdaBuilder(const IndexLambdaBuilder&) = delete;
  IndexLambdaBuilder(IndexLambdaBuilder&&) = delete;

  AnfExpr IndexLambda(
      const std::vector<std::string>& in_shape_vars,
      const std::vector<std::string>& in_data_vars,
      const std::vector<std::string>& out_shape_vars,
      const std::vector<std::string>& out_data_vars,
      const std::vector<std::string>& index_vars,
      const std::function<AnfExpr(IndexLambdaBuildContext&)> GetBody) {
    PADDLE_ENFORCE_EQ(
        in_shape_vars.size(),
        in_data_vars.size(),
        phi::errors::InvalidArgument("in_shape_vars.size() should equal to "
                                     "in_data_vars.size(). (%s v.s. %s)",
                                     in_shape_vars.size(),
                                     in_data_vars.size()));
    PADDLE_ENFORCE_EQ(
        out_shape_vars.size(),
        out_data_vars.size(),
        phi::errors::InvalidArgument("out_shape_vars.size() should equal to "
                                     "out_data_vars.size(). (%s v.s. %s)",
                                     out_shape_vars.size(),
                                     out_data_vars.size()));

    return lmbd_builder_.NestedLambda(
        {in_shape_vars,
         in_data_vars,
         out_shape_vars,
         out_data_vars,
         index_vars},
        [&](auto& let_context) {
          IndexLambdaBuildContext ctx(&let_context);
          return GetBody(ctx);
        });
  }

 private:
  LambdaExprBuilder lmbd_builder_;
};

}  // namespace pexpr
