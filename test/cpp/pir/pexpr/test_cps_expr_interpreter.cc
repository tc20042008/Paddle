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
#include "paddle/pir/include/dialect/pexpr/anf_expr_util.h"
#include "paddle/pir/include/dialect/pexpr/cps_expr_interpreter.h"
#include "paddle/pir/include/dialect/pexpr/lambda_expr_builder.h"
#include "test/cpp/pir/pexpr/test_value.h"

namespace pexpr::tests {

TEST(CpsExprInterpreter, simple) {
  LambdaExprBuilder lmbd{};
  AnfExpr lmbd_expr = lmbd.Lambda({}, [](auto& ctx) { return ctx.Int64(1); });
  const auto& core_expr = ConvertAnfExprToCoreExpr(lmbd_expr);
  ASSERT_TRUE(core_expr.Has<Atomic<CoreExpr>>());
  const auto& atomic = core_expr.Get<Atomic<CoreExpr>>();
  ASSERT_TRUE(atomic.Has<Lambda<CoreExpr>>());
  const auto& lambda = atomic.Get<Lambda<CoreExpr>>();
  CpsExprInterpreter<Val> interpreter{};
  Closure<Val> closure{lambda,
                       interpreter.env_mgr()->New(interpreter.builtin_frame())};
  const auto& interpret_ret =
      interpreter.Interpret(closure, std::vector<Val>{});
  if (!interpret_ret.HasOkValue()) {
    LOG(ERROR) << "error-type: " << interpret_ret.GetError().class_name()
               << ", error-msg: " << interpret_ret.GetError().msg();
  }
  ASSERT_TRUE(interpret_ret.HasOkValue());
  const auto& val = interpret_ret.GetOkValue();
  ASSERT_TRUE(val.Has<int64_t>());
  ASSERT_EQ(val.Get<int64_t>(), 1);
}

TEST(CpsExprInterpreter, lambda) {
  LambdaExprBuilder lmbd{};
  AnfExpr lmbd_expr = lmbd.Lambda({}, [&](auto& ctx) {
    ctx.Var("identity") = LambdaExprBuilder{}.Lambda(
        {"x"}, [&](auto& inner_ctx) { return inner_ctx.Var("x"); });
    return ctx.Var("identity").Call(ctx.Int64(1));
  });
  const auto& core_expr = ConvertAnfExprToCoreExpr(lmbd_expr);
  ASSERT_TRUE(core_expr.Has<Atomic<CoreExpr>>());
  const auto& atomic = core_expr.Get<Atomic<CoreExpr>>();
  ASSERT_TRUE(atomic.Has<Lambda<CoreExpr>>());
  const auto& lambda = atomic.Get<Lambda<CoreExpr>>();
  CpsExprInterpreter<Val> interpreter{};
  Closure<Val> closure{lambda,
                       interpreter.env_mgr()->New(interpreter.builtin_frame())};
  const auto& interpret_ret =
      interpreter.Interpret(closure, std::vector<Val>{});
  if (!interpret_ret.HasOkValue()) {
    LOG(ERROR) << "error-type: " << interpret_ret.GetError().class_name()
               << ", error-msg: " << interpret_ret.GetError().msg();
  }
  ASSERT_TRUE(interpret_ret.HasOkValue());
  const auto& val = interpret_ret.GetOkValue();
  ASSERT_TRUE(val.Has<int64_t>());
  ASSERT_EQ(val.Get<int64_t>(), 1);
}

}  // namespace pexpr::tests
