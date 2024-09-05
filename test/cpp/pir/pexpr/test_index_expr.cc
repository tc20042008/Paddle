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

#include <ctime>
#include <thread>  // NOLINT
#include "gtest/gtest.h"

#include "paddle/common/errors.h"
#include "paddle/pir/include/dialect/pexpr/anf_expr_builder.h"
#include "paddle/pir/include/dialect/pexpr/anf_expr_util.h"
#include "paddle/pir/include/dialect/pexpr/core_expr_builder.h"
#include "paddle/pir/include/dialect/pexpr/index_expr_interpreter.h"
#include "paddle/pir/include/dialect/pexpr/lambda_expr_builder.h"

namespace pexpr::index_expr::tests {

TEST(IndexExpr, kNothingIndexTupleExpr) {
  size_t seq_no0 = 0;
  const auto& GenSeqNo0 = [&]() { return seq_no0++; };
  LambdaExprBuilder lmbd(GenSeqNo0);
  AnfExpr lmbd_expr = lmbd.Lambda(
      {}, [](auto& ctx) { return ctx.Var("kNothingIndexTupleExpr"); });
  CoreExpr core_expr = ConvertAnfExprToCoreExpr(lmbd_expr);
  ASSERT_TRUE((core_expr.Has<Atomic<CoreExpr>>()));
  ASSERT_TRUE((core_expr.Get<Atomic<CoreExpr>>().Has<Lambda<CoreExpr>>()));
  const auto& lambda =
      core_expr.Get<Atomic<CoreExpr>>().Get<Lambda<CoreExpr>>();
  index_expr::IndexExprInterpreter interpreter;
  Result<index_expr::Val> result = interpreter(lambda, {});
  ASSERT_TRUE(result.Has<index_expr::Val>());
  const auto& ret = result.Get<index_expr::Val>();
  ASSERT_TRUE((ret.Has<IndexExprValue>()));
  ASSERT_TRUE((ret.Get<IndexExprValue>().Has<IndexTupleExpr>()));
  const auto& index_tuple_expr =
      ret.Get<IndexExprValue>().Get<IndexTupleExpr>();
  ASSERT_TRUE((index_tuple_expr.Has<NothingIndexTupleExpr>()));
}

TEST(IndexExpr, IndexTupleExprReshape) {
  size_t seq_no0 = 0;
  const auto& GenSeqNo0 = [&]() { return seq_no0++; };
  LambdaExprBuilder lmbd(GenSeqNo0);
  AnfExpr lmbd_expr = lmbd.Lambda({"expr"}, [](auto& ctx) {
    const auto& dim_expr =
        ctx.Call("list", ctx.Int64(2), ctx.Int64(3), ctx.Int64(4));
    return ctx.Call("IndexTupleExprReshape", dim_expr, ctx.Var("expr"));
  });
  CoreExpr core_expr = ConvertAnfExprToCoreExpr(lmbd_expr);
  ASSERT_TRUE((core_expr.Has<Atomic<CoreExpr>>()));
  ASSERT_TRUE((core_expr.Get<Atomic<CoreExpr>>().Has<Lambda<CoreExpr>>()));
  const auto& lambda =
      core_expr.Get<Atomic<CoreExpr>>().Get<Lambda<CoreExpr>>();
  index_expr::IndexExprInterpreter interpreter;
  Result<index_expr::Val> result = interpreter(
      lambda,
      {index_expr::Val{IndexExprValue{
          IndexTupleExpr{IndexTupleExprDomain{adt::List<symbol::DimExpr>{
              symbol::DimExpr{6}, symbol::DimExpr{4}}}}}}});
  if (!result.HasOkValue()) {
    LOG(ERROR) << "error-type: " << result.GetError().class_name()
               << ", error-msg: " << result.GetError().msg();
  }
  ASSERT_TRUE(result.HasOkValue());
  const auto& ret = result.GetOkValue();
  ASSERT_TRUE((ret.Has<IndexExprValue>()));
  ASSERT_TRUE((ret.Get<IndexExprValue>().Has<IndexTupleExpr>()));
  const auto& index_tuple_expr =
      ret.Get<IndexExprValue>().Get<IndexTupleExpr>();
  ASSERT_TRUE((index_tuple_expr.Has<IndexTupleExprReshape<IndexTupleExpr>>()));
}

TEST(IndexExpr, IndexTupleExprReshape_failed) {
  size_t seq_no0 = 0;
  const auto& GenSeqNo0 = [&]() { return seq_no0++; };
  LambdaExprBuilder lmbd(GenSeqNo0);
  AnfExpr lmbd_expr = lmbd.Lambda({"expr"}, [](auto& ctx) {
    const auto& dim_expr =
        ctx.Call("list", ctx.Int64(2), ctx.Int64(3), ctx.Int64(4));
    return ctx.Call("IndexTupleExprReshape", dim_expr, ctx.Var("expr"));
  });
  CoreExpr core_expr = ConvertAnfExprToCoreExpr(lmbd_expr);
  ASSERT_TRUE((core_expr.Has<Atomic<CoreExpr>>()));
  ASSERT_TRUE((core_expr.Get<Atomic<CoreExpr>>().Has<Lambda<CoreExpr>>()));
  const auto& lambda =
      core_expr.Get<Atomic<CoreExpr>>().Get<Lambda<CoreExpr>>();
  index_expr::IndexExprInterpreter interpreter;
  Result<index_expr::Val> result = interpreter(
      lambda,
      {index_expr::Val{IndexExprValue{
          IndexTupleExpr{IndexTupleExprDomain{adt::List<symbol::DimExpr>{
              symbol::DimExpr{5}, symbol::DimExpr{4}}}}}}});
  ASSERT_TRUE(result.Has<Error>());
}
}  // namespace pexpr::index_expr::tests
