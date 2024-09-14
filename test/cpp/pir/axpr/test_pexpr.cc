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

#include "ap/axpr/anf_expr_builder.h"
#include "ap/axpr/anf_expr_util.h"
#include "ap/axpr/core_expr_builder.h"
#include "ap/axpr/lambda_expr_builder.h"
#include "paddle/common/errors.h"

namespace ap::axpr::tests {

TEST(CoreExpr, operator_eq_0) {
  CoreExprBuilder core{};
  using Var = tVar<std::string>;
  CoreExpr lhs = core.ComposedCallAtomic(
      core.Lambda({Var{"c"}},
                  core.ComposedCallAtomic(core.Var(kBuiltinId()),
                                          core.Var(kBuiltinList()),
                                          {
                                              core.Var("a"),
                                              core.Var("b"),
                                              core.Var("c"),
                                          })),
      core.Var("op2"),
      {});
  CoreExpr rhs = core.ComposedCallAtomic(core.Var(kBuiltinId()),
                                         core.Var(kBuiltinList()),
                                         {
                                             core.Var("a"),
                                             core.Var("a"),
                                             core.Var("c"),
                                         });
  ASSERT_FALSE(lhs == rhs);
}

TEST(CoreExpr, operator_eq_1) {
  CoreExprBuilder core{};
  using Var = tVar<std::string>;
  CoreExpr lhs = core.ComposedCallAtomic(
      core.Lambda(
          {Var{"b"}},
          core.ComposedCallAtomic(
              core.Lambda({Var{"c"}},
                          core.ComposedCallAtomic(core.Var(kBuiltinId()),
                                                  core.Var(kBuiltinList()),
                                                  {
                                                      core.Var("a"),
                                                      core.Var("b"),
                                                      core.Var("c"),
                                                  })),
              core.Var("op2"),
              {})),
      core.Var("op1"),
      {});
  CoreExpr rhs = core.ComposedCallAtomic(
      core.Lambda({Var{"c"}},
                  core.ComposedCallAtomic(core.Var(kBuiltinId()),
                                          core.Var(kBuiltinList()),
                                          {
                                              core.Var("a"),
                                              core.Var("a"),
                                              core.Var("c"),
                                          })),
      core.Var("op2"),
      {});
  ASSERT_FALSE(lhs == rhs);
}

TEST(CoreExpr, ConvertAnfExprToCoreExpr) {
  auto anf = AnfExprBuilder();
  AnfExpr anf_expr = anf.Let(
      {
          anf.Bind("a", anf.Call(anf.Var("op0"), {})),
          anf.Bind("b", anf.Call(anf.Var("op1"), {})),
          anf.Bind("c", anf.Call(anf.Var("op2"), {})),
      },
      anf.Call(anf.Var("list"), {anf.Var("a"), anf.Var("b"), anf.Var("c")}));
  const auto& opt_anf_expr =
      MakeAnfExprFromJsonString(anf_expr.DumpToJsonString());
  ASSERT_TRUE(opt_anf_expr.HasOkValue());
  anf_expr = opt_anf_expr.GetOkValue();
  CoreExpr core_expr = ConvertAnfExprToCoreExpr(anf_expr);
  CoreExprBuilder core{};
  using Var = tVar<std::string>;
  CoreExpr expected = core.ComposedCallAtomic(
      core.Lambda(
          {Var{"a"}},
          core.ComposedCallAtomic(
              core.Lambda({Var{"b"}},
                          core.ComposedCallAtomic(
                              core.Lambda({Var{"c"}},
                                          core.ComposedCallAtomic(
                                              core.Var(kBuiltinReturn()),
                                              core.Var("list"),
                                              {
                                                  core.Var("a"),
                                                  core.Var("b"),
                                                  core.Var("c"),
                                              })),
                              core.Var("op2"),
                              {})),
              core.Var("op1"),
              {})),
      core.Var("op0"),
      {});
  ASSERT_EQ(core_expr, expected);
}

TEST(LambdaExprBuilder, Let) {
  auto anf = AnfExprBuilder();
  AnfExpr anf_expr = anf.Let(
      {
          anf.Bind("a", anf.Call(anf.Var("op0"), {})),
          anf.Bind("b", anf.Call(anf.Var("op1"), {})),
      },
      anf.Call(anf.Var(kBuiltinList()), {anf.Var("a"), anf.Var("b")}));
  size_t seq_no = 0;
  const auto& GenSeqNo = [&]() { return seq_no++; };
  LambdaExprBuilder lmbd(GenSeqNo);
  AnfExpr lmbd_expr = lmbd.Let([](auto& ctx) {
    ctx.Var("a") = ctx.Call("op0");
    ctx.Var("b") = ctx.Call("op1");
    return ctx.Call(kBuiltinList(), ctx.Var("a"), ctx.Var("b"));
  });
  const auto& core_expr = ConvertAnfExprToCoreExpr(lmbd_expr);
  const auto& expected = ConvertAnfExprToCoreExpr(anf_expr);
  ASSERT_EQ(core_expr, expected);
}

TEST(LambdaExprBuilder, LetTmpVar) {
  auto anf = AnfExprBuilder();
  AnfExpr anf_expr = anf.Let(
      {
          anf.Bind("__lambda_expr_tmp0", anf.Call(anf.Var("op0"), {})),
      },
      anf.Call(anf.Var(kBuiltinList()), {anf.Var("__lambda_expr_tmp0")}));
  size_t seq_no = 0;
  const auto& GenSeqNo = [&]() { return seq_no++; };
  LambdaExprBuilder lmbd(GenSeqNo);
  AnfExpr lmbd_expr = lmbd.Let(
      [](auto& ctx) { return ctx.Call(kBuiltinList(), ctx.Call("op0")); });
  const auto& core_expr = ConvertAnfExprToCoreExpr(lmbd_expr);
  const auto& expected = ConvertAnfExprToCoreExpr(anf_expr);
  ASSERT_EQ(core_expr, expected);
}

TEST(LambdaExprBuilder, Lambda) {
  auto anf = AnfExprBuilder();
  AnfExpr anf_expr = anf.Let(
      {
          anf.Bind("__lambda_expr_tmp0", anf.Call(anf.Var("op0"), {})),
      },
      anf.Call(anf.Var(kBuiltinList()), {anf.Var("__lambda_expr_tmp0")}));
  size_t seq_no0 = 0;
  const auto& GenSeqNo0 = [&]() { return seq_no0++; };
  LambdaExprBuilder lmbd(GenSeqNo0);
  AnfExpr lmbd_expr = lmbd.Let(
      [](auto& ctx) { return ctx.Call(kBuiltinList(), ctx.Call("op0")); });
  const auto& core_expr = ConvertAnfExprToCoreExpr(lmbd_expr);
  const auto& expected = ConvertAnfExprToCoreExpr(anf_expr);
  ASSERT_EQ(core_expr, expected);
}

}  // namespace ap::axpr::tests
