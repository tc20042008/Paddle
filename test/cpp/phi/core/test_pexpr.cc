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
#include "paddle/phi/core/pexpr/cps_builder.h"
#include "paddle/phi/core/pexpr/pexpr_util.h"

namespace pexpr::tests {

TEST(CpsExpr, ConvertAnfExprToCpsExpr) {
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
  CpsExpr cps_expr = ConvertAnfExprToCpsExpr(anf_expr);
  CpsExprBuilder cps{};
  using Var = tVar<std::string>;
  CpsExpr expected = cps.Lambda(
      {Var{"return"}},
      cps.Call(
          cps.Var("op0"),
          {cps.Lambda(
              {Var{"a"}},
              cps.Call(cps.Var("op1"),
                       {cps.Lambda(
                           {Var{"b"}},
                           cps.Call(cps.Var("op2"),
                                    {cps.Lambda({Var{"c"}},
                                                cps.Call(cps.Var("list"),
                                                         {
                                                             cps.Var("a"),
                                                             cps.Var("b"),
                                                             cps.Var("c"),
                                                             cps.Var("return"),
                                                         }))}))}))}));
  ASSERT_EQ(cps_expr, expected);
}

}  // namespace pexpr::tests
