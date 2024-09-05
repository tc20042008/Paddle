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
  const auto& interpret_ret = interpreter.Interpret(lambda, std::vector<Val>{});
  if (!interpret_ret.HasOkValue()) {
    LOG(ERROR) << "error-type: " << interpret_ret.GetError().class_name()
               << ", error-msg: " << interpret_ret.GetError().msg();
  }
  ASSERT_TRUE(interpret_ret.HasOkValue());
  const auto& val = interpret_ret.GetOkValue();
  const auto& int_val = CastToArithmeticValue<int64_t>(val);
  ASSERT_TRUE(int_val.HasOkValue());
  ASSERT_EQ(int_val.GetOkValue(), 1);
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
  const auto& interpret_ret = interpreter.Interpret(lambda, std::vector<Val>{});
  if (!interpret_ret.HasOkValue()) {
    LOG(ERROR) << "error-type: " << interpret_ret.GetError().class_name()
               << ", error-msg: " << interpret_ret.GetError().msg();
  }
  ASSERT_TRUE(interpret_ret.HasOkValue());
  const auto& val = interpret_ret.GetOkValue();
  const auto& int_val = CastToArithmeticValue<int64_t>(val);
  ASSERT_TRUE(int_val.HasOkValue());
  ASSERT_EQ(int_val.GetOkValue(), 1);
}

TEST(CpsExprInterpreter, arithmetic_value) {
  const std::string json_str = R"(
    [
      "lambda",
      [
        "a",
        "b"
      ],
      [
        "__builtin_let__",
        [
          [
            "__lambda_expr_tmp0",
            [
              "__builtin_Add__",
              "a",
              "b"
            ]
          ],
          [
            "x0",
            [
              "__builtin_identity__",
              "__lambda_expr_tmp0"
            ]
          ],
          [
            "__lambda_expr_tmp1",
            [
              "__builtin_Mul__",
              "x0",
              "a"
            ]
          ],
          [
            "x1",
            [
              "__builtin_identity__",
              "__lambda_expr_tmp1"
            ]
          ],
          [
            "__lambda_expr_tmp2",
            [
              "__builtin_Div__",
              "x1",
              "b"
            ]
          ],
          [
            "x2",
            [
              "__builtin_identity__",
              "__lambda_expr_tmp2"
            ]
          ],
          [
            "__lambda_expr_tmp3",
            [
              "__builtin_Mod__",
              "a",
              "b"
            ]
          ],
          [
            "x3",
            [
              "__builtin_identity__",
              "__lambda_expr_tmp3"
            ]
          ],
          [
            "__lambda_expr_tmp4",
            [
              "__builtin_Neg__",
              "x2"
            ]
          ],
          [
            "x4",
            [
              "__builtin_identity__",
              "__lambda_expr_tmp4"
            ]
          ],
          [
            "__lambda_expr_tmp5",
            [
              "__builtin_EQ__",
              "x0",
              1
            ]
          ],
          [
            "b0",
            [
              "__builtin_identity__",
              "__lambda_expr_tmp5"
            ]
          ],
          [
            "__lambda_expr_tmp6",
            [
              "__builtin_NE__",
              "x2",
              "x3"
            ]
          ],
          [
            "b1",
            [
              "__builtin_identity__",
              "__lambda_expr_tmp6"
            ]
          ],
          [
            "__lambda_expr_tmp7",
            [
              "__builtin_GT__",
              "x0",
              "x1"
            ]
          ],
          [
            "b2",
            [
              "__builtin_identity__",
              "__lambda_expr_tmp7"
            ]
          ],
          [
            "__lambda_expr_tmp8",
            [
              "__builtin_GE__",
              "x0",
              "x1"
            ]
          ],
          [
            "b3",
            [
              "__builtin_identity__",
              "__lambda_expr_tmp8"
            ]
          ],
          [
            "__lambda_expr_tmp9",
            [
              "__builtin_LT__",
              "x0",
              "x1"
            ]
          ],
          [
            "b4",
            [
              "__builtin_identity__",
              "__lambda_expr_tmp9"
            ]
          ],
          [
            "__lambda_expr_tmp10",
            [
              "__builtin_LE__",
              "x0",
              "x1"
            ]
          ],
          [
            "b5",
            [
              "__builtin_identity__",
              "__lambda_expr_tmp10"
            ]
          ],
          [
            "__lambda_expr_tmp11",
            [
              "__builtin_Not__",
              "b3"
            ]
          ],
          [
            "b6",
            [
              "__builtin_identity__",
              "__lambda_expr_tmp11"
            ]
          ]
        ],
        [
          "__builtin_identity__",
          "x0"
        ]
      ]
    ]
  )";
  const auto& anf_expr = pexpr::MakeAnfExprFromJsonString(json_str);
  if (anf_expr.HasError()) {
    LOG(ERROR) << "error-type: " << anf_expr.GetError().class_name()
               << ", error-msg: " << anf_expr.GetError().msg();
  }
  ASSERT_TRUE(anf_expr.HasOkValue());
  const auto& core_expr = ConvertAnfExprToCoreExpr(anf_expr.GetOkValue());
  ASSERT_TRUE(core_expr.Has<Atomic<CoreExpr>>());
  const auto& atomic = core_expr.Get<Atomic<CoreExpr>>();
  ASSERT_TRUE(atomic.Has<Lambda<CoreExpr>>());
  const auto& lambda = atomic.Get<Lambda<CoreExpr>>();
  CpsExprInterpreter<Val> interpreter{};
  ArithmeticValue x{int64_t(2)};
  ArithmeticValue y{int32_t(3)};
  const auto& interpret_ret = interpreter.Interpret(lambda, {x, y});
  if (!interpret_ret.HasOkValue()) {
    LOG(ERROR) << "error-type: " << interpret_ret.GetError().class_name()
               << ", error-msg: " << interpret_ret.GetError().msg();
  }
  ASSERT_TRUE(interpret_ret.HasOkValue());
  const auto& val = interpret_ret.GetOkValue();
  const auto& int_val = CastToArithmeticValue<int64_t>(val);
  ASSERT_TRUE(int_val.HasOkValue());
  ASSERT_EQ(int_val.GetOkValue(), 5);
}

}  // namespace pexpr::tests
