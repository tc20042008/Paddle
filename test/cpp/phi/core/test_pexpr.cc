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
#include "paddle/phi/core/pexpr/anf_expr_builder.h"
#include "paddle/phi/core/pexpr/anf_expr_util.h"
#include "paddle/phi/core/pexpr/core_expr_builder.h"
#include "paddle/phi/core/pexpr/core_expr_util.h"
#include "paddle/phi/core/pexpr/lambda_expr_builder.h"

namespace pexpr::tests {

TEST(CoreExpr, operator_eq_0) {
  CoreExprBuilder core{};
  using Var = tVar<std::string>;
  CoreExpr lhs =
      core.ComposedCall(core.Lambda({Var{"c"}},
                                    core.ComposedCall(core.Var(kBuiltinId),
                                                      core.Var("list"),
                                                      {
                                                          core.Var("a"),
                                                          core.Var("b"),
                                                          core.Var("c"),
                                                      })),
                        core.Var("op2"),
                        {});
  CoreExpr rhs = core.ComposedCall(core.Var(kBuiltinId),
                                   core.Var("list"),
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
  CoreExpr lhs = core.ComposedCall(
      core.Lambda(
          {Var{"b"}},
          core.ComposedCall(core.Lambda({Var{"c"}},
                                        core.ComposedCall(core.Var(kBuiltinId),
                                                          core.Var("list"),
                                                          {
                                                              core.Var("a"),
                                                              core.Var("b"),
                                                              core.Var("c"),
                                                          })),
                            core.Var("op2"),
                            {})),
      core.Var("op1"),
      {});
  CoreExpr rhs =
      core.ComposedCall(core.Lambda({Var{"c"}},
                                    core.ComposedCall(core.Var(kBuiltinId),
                                                      core.Var("list"),
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
      AnfExpr::ParseFromJsonString(anf_expr.DumpToJsonString());
  ASSERT_TRUE(opt_anf_expr.has_value());
  anf_expr = opt_anf_expr.value();
  CoreExpr core_expr = ConvertAnfExprToCoreExpr(anf_expr);
  const auto& opt_core_expr =
      CoreExpr::ParseFromJsonString(core_expr.DumpToJsonString());
  ASSERT_TRUE(opt_core_expr.has_value());
  core_expr = opt_core_expr.value();
  CoreExprBuilder core{};
  using Var = tVar<std::string>;
  CoreExpr expected = core.ComposedCall(
      core.Lambda({Var{"a"}},
                  core.ComposedCall(
                      core.Lambda({Var{"b"}},
                                  core.ComposedCall(
                                      core.Lambda({Var{"c"}},
                                                  core.ComposedCall(
                                                      core.Var(kBuiltinId),
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

TEST(CoreExpr, InlineBuiltinId) {
  auto anf = AnfExprBuilder();
  AnfExpr anf_expr = anf.Let(
      {
          anf.Bind("a", anf.Call(anf.Var("op0"), {})),
          anf.Bind("b", anf.Call(anf.Var(kBuiltinId), {anf.Var("a")})),
          anf.Bind("c", anf.Call(anf.Var("op2"), {})),
      },
      anf.Call(anf.Var("list"), {anf.Var("a"), anf.Var("b"), anf.Var("c")}));
  CoreExpr core_expr = ConvertAnfExprToCoreExpr(anf_expr);
  core_expr = Inline(core_expr);
  CoreExprBuilder core{};
  using Var = tVar<std::string>;
  CoreExpr expected = core.ComposedCall(
      core.Lambda(
          {Var{"a"}},
          core.ComposedCall(core.Lambda({Var{"c"}},
                                        core.ComposedCall(core.Var(kBuiltinId),
                                                          core.Var("list"),
                                                          {
                                                              core.Var("a"),
                                                              core.Var("a"),
                                                              core.Var("c"),
                                                          })),
                            core.Var("op2"),
                            {})),
      core.Var("op0"),
      {});
  ASSERT_EQ(core_expr, expected);
}

TEST(CoreExpr, InlineInnerLambda) {
  using Var = tVar<std::string>;
  auto anf = AnfExprBuilder();
  AnfExpr anf_expr = anf.Let(
      {
          anf.Bind("a", anf.Call(anf.Var("op0"), {})),
          anf.Bind(
              "b",
              anf.Call(anf.Lambda({Var{"tmp"}}, anf.Int64(0)), {anf.Var("a")})),
          anf.Bind("c", anf.Call(anf.Var("op2"), {})),
      },
      anf.Call(anf.Var("list"), {anf.Var("a"), anf.Var("b"), anf.Var("c")}));
  CoreExpr core_expr = ConvertAnfExprToCoreExpr(anf_expr);
  core_expr = Inline(core_expr);
  CoreExprBuilder core{};
  CoreExpr expected = core.ComposedCall(
      core.Lambda({Var{"a"}},
                  core.ComposedCall(
                      core.Lambda({Var{"b"}},
                                  core.ComposedCall(
                                      core.Lambda({Var{"c"}},
                                                  core.ComposedCall(
                                                      core.Var(kBuiltinId),
                                                      core.Var("list"),
                                                      {
                                                          core.Var("a"),
                                                          core.Var("b"),
                                                          core.Var("c"),
                                                      })),
                                      core.Var("op2"),
                                      {})),
                      core.Lambda({},
                                  core.ComposedCall(core.Var(kBuiltinId),
                                                    core.Var(kBuiltinId),
                                                    {core.Int64(0)})),
                      {})),
      core.Var("op0"),
      {});
  ASSERT_EQ(core_expr, expected);
}

TEST(CoreExpr, ReplaceLambdaArgName) {
  using Var = tVar<std::string>;
  CoreExprBuilder core{};
  CoreExpr core_expr = core.ComposedCall(
      core.Lambda({Var{"a"}},
                  core.ComposedCall(
                      core.Lambda({Var{"b"}},
                                  core.ComposedCall(
                                      core.Lambda({Var{"c"}},
                                                  core.ComposedCall(
                                                      core.Var(kBuiltinId),
                                                      core.Var("list"),
                                                      {
                                                          core.Var("a"),
                                                          core.Var("b"),
                                                          core.Var("c"),
                                                      })),
                                      core.Var("op2"),
                                      {})),
                      core.Lambda({},
                                  core.ComposedCall(core.Var(kBuiltinId),
                                                    core.Var(kBuiltinId),
                                                    {core.Int64(0)})),
                      {})),
      core.Var("op0"),
      {});
  CoreExpr replaced =
      ReplaceLambdaArgName(core_expr, "c", []() { return std::string("d"); });

  CoreExpr expected = core.ComposedCall(
      core.Lambda({Var{"a"}},
                  core.ComposedCall(
                      core.Lambda({Var{"b"}},
                                  core.ComposedCall(
                                      core.Lambda({Var{"d"}},
                                                  core.ComposedCall(
                                                      core.Var(kBuiltinId),
                                                      core.Var("list"),
                                                      {
                                                          core.Var("a"),
                                                          core.Var("b"),
                                                          core.Var("d"),
                                                      })),
                                      core.Var("op2"),
                                      {})),
                      core.Lambda({},
                                  core.ComposedCall(core.Var(kBuiltinId),
                                                    core.Var(kBuiltinId),
                                                    {core.Int64(0)})),
                      {})),
      core.Var("op0"),
      {});
  ASSERT_EQ(replaced, expected);
}

TEST(LambdaExprBuilder, Let) {
  auto anf = AnfExprBuilder();
  AnfExpr anf_expr = anf.Let(
      {
          anf.Bind("a", anf.Call(anf.Var("op0"), {})),
          anf.Bind("b", anf.Call(anf.Var("op1"), {})),
      },
      anf.Call(anf.Var("list"), {anf.Var("a"), anf.Var("b")}));
  LambdaExprBuilder lmbd;
  AnfExpr lmbd_expr = lmbd.Let([](auto& ctx) {
    ctx.Var("a") = ctx.Call("op0");
    ctx.Var("b") = ctx.Call("op1");
    return ctx.Call("list", ctx.Var("a"), ctx.Var("b"));
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
      anf.Call(anf.Var("list"), {anf.Var("__lambda_expr_tmp0")}));
  LambdaExprBuilder lmbd;
  AnfExpr lmbd_expr =
      lmbd.Let([](auto& ctx) { return ctx.Call("list", ctx.Call("op0")); });
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
      anf.Call(anf.Var("list"), {anf.Var("__lambda_expr_tmp0")}));
  LambdaExprBuilder lmbd;
  AnfExpr lmbd_expr =
      lmbd.Let([](auto& ctx) { return ctx.Call("list", ctx.Call("op0")); });
  const auto& core_expr = ConvertAnfExprToCoreExpr(lmbd_expr);
  const auto& expected = ConvertAnfExprToCoreExpr(anf_expr);
  ASSERT_EQ(core_expr, expected);
}

}  // namespace pexpr::tests
