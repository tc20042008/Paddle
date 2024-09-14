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

#include "ap/axpr/anf_expr_util.h"
#include "ap/axpr/const_std_vector_ptr.h"
#include "ap/axpr/const_std_vector_ptr_method_class.h"
#include "ap/axpr/cps_expr_interpreter.h"
#include "ap/axpr/lambda_expr_builder.h"
#include "ap/axpr/value_method_class.h"
#include "paddle/common/errors.h"

namespace ap::axpr::tests {

namespace {

template <typename ValueT>
using ValueImpl = ValueBase<ValueT, const std::vector<int>*>;

struct TestValue : public ValueImpl<TestValue> {
  using ValueImpl<TestValue>::ValueImpl;
  DEFINE_ADT_VARIANT_METHODS(ValueImpl<TestValue>);
};

using Val = TestValue;

}  // namespace

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
  const auto& int_val = MethodClass<Val>::TryGet<int64_t>(val);
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
  const auto& int_val = MethodClass<Val>::TryGet<int64_t>(val);
  ASSERT_TRUE(int_val.HasOkValue());
  ASSERT_EQ(int_val.GetOkValue(), 1);
}

TEST(CpsExprInterpreter, data_value) {
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
  const auto& anf_expr = ap::axpr::MakeAnfExprFromJsonString(json_str);
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
  int64_t x = 3;
  int64_t y = 5;
  const auto& interpret_ret = interpreter.Interpret(lambda, {x, y});
  if (!interpret_ret.HasOkValue()) {
    LOG(ERROR) << interpret_ret.GetError().class_name() << ": "
               << interpret_ret.GetError().msg();
  }
  ASSERT_TRUE(interpret_ret.HasOkValue());
  const auto& val = interpret_ret.GetOkValue();
  const auto& int_val = MethodClass<Val>::TryGet<int64_t>(val);
  ASSERT_TRUE(int_val.HasOkValue());
  ASSERT_EQ(int_val.GetOkValue(), 8);
}

TEST(CpsExprInterpreter, vector_getitem) {
  const std::string json_str = R"(
    [
      "lambda",
      [
        "vect"
      ],
      [
        "__builtin_let__",
        [
          [
            "__lambda_expr_tmp0",
            [
              "__builtin_getitem__",
              "vect",
              0
            ]
          ],
          [
            "__lambda_expr_tmp1",
            [
              "__builtin_return__",
              "__lambda_expr_tmp0"
            ]
          ]
        ],
        [
          "__builtin_identity__",
          "__lambda_expr_tmp1"
        ]
      ]
    ]
  )";
  const auto& anf_expr = ap::axpr::MakeAnfExprFromJsonString(json_str);
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
  const std::vector<int> vect{666, 888};
  const auto& interpret_ret = interpreter.Interpret(lambda, {&vect});
  if (!interpret_ret.HasOkValue()) {
    LOG(ERROR) << interpret_ret.GetError().class_name() << ": "
               << interpret_ret.GetError().msg();
  }
  ASSERT_TRUE(interpret_ret.HasOkValue());
  const auto& val = interpret_ret.GetOkValue();
  const auto& int_val = MethodClass<Val>::TryGet<int64_t>(val);
  ASSERT_TRUE(int_val.HasOkValue());
  ASSERT_EQ(int_val.GetOkValue(), 666);
}

TEST(CpsExprInterpreter, test_float) {
  const std::string json_str = R"(
    [
      "lambda",
      [
        "v"
      ],
      [
        "__builtin_let__",
        [
          [
            "__lambda_expr_tmp0",
            [
              "__builtin_Add__",
              "v",
              1.0
            ]
          ],
          [
            "__lambda_expr_tmp1",
            [
              "__builtin_return__",
              "__lambda_expr_tmp0"
            ]
          ]
        ],
        [
          "__builtin_identity__",
          "__lambda_expr_tmp1"
        ]
      ]
    ]
  )";
  const auto& anf_expr = ap::axpr::MakeAnfExprFromJsonString(json_str);
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
  double float_val = -1;
  const auto& interpret_ret = interpreter.Interpret(lambda, {float_val});
  if (!interpret_ret.HasOkValue()) {
    LOG(ERROR) << interpret_ret.GetError().class_name() << ": "
               << interpret_ret.GetError().msg();
  }
  ASSERT_TRUE(interpret_ret.HasOkValue());
  const auto& val = interpret_ret.GetOkValue();
  const auto& int_val = MethodClass<Val>::TryGet<double>(val);
  ASSERT_TRUE(int_val.HasOkValue());
  ASSERT_EQ(int_val.GetOkValue(), 0.0);
}

}  // namespace ap::axpr::tests
