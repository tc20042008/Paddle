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
#include "paddle/phi/core/pexpr/anf_builder.h"
#include "paddle/phi/core/pexpr/core_expr_builder.h"
#include "paddle/phi/core/pexpr/core_expr_util.h"

namespace pexpr::tests {

TEST(pexpr, ConvertAnfExprToCoreExpr) {
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
      core.Lambda(
          {Var{"a"}},
          core.ComposedCall(
              core.Lambda({Var{"b"}},
                          core.ComposedCall(
                              core.Lambda({Var{"c"}},
                                          core.ComposedCall(
                                              core.Var("__builtin_identity__"),
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
  LOG(ERROR) << "\n" << core_expr;
}

}  // namespace pexpr::tests
