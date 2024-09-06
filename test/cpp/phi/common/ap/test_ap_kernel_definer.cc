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

#include <complex>
#include <sstream>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/phi/common/ap/define_ctx_value.h"
#include "paddle/phi/common/ap/define_ctx_value_method_class.h"
#include "paddle/pir/include/dialect/pexpr/anf_expr_util.h"
#include "paddle/pir/include/dialect/pexpr/cast_util.h"
#include "paddle/pir/include/dialect/pexpr/core_expr.h"
#include "paddle/pir/include/dialect/pexpr/cps_expr_interpreter.h"
#include "paddle/pir/include/dialect/pexpr/lambda_expr_builder.h"

namespace ap::kernel_define::test {

TEST(KernelDefine, ArgType) {
  pexpr::LambdaExprBuilder lmbd;
  pexpr::AnfExpr anf_expr = lmbd.Lambda({"ctx"}, [&](auto& ctx) {
    return ctx.Var("ctx").Attr("const_int32_ptr");
  });
  pexpr::CoreExpr core_expr = pexpr::ConvertAnfExprToCoreExpr(anf_expr);
  ASSERT_TRUE(core_expr.Has<pexpr::Atomic<pexpr::CoreExpr>>());
  const auto& atomic = core_expr.Get<pexpr::Atomic<pexpr::CoreExpr>>();
  ASSERT_TRUE(atomic.Has<pexpr::Lambda<pexpr::CoreExpr>>());
  const auto& lambda = atomic.Get<pexpr::Lambda<pexpr::CoreExpr>>();
  pexpr::CpsExprInterpreter<Val> interpreter;
  DefinerCtx<Val> ctx{DefinerRawCtx{}, pexpr::Object<Val>{}};
  const Result<Val>& ret = interpreter.Interpret(lambda, {ctx});
  if (ret.HasError()) {
    LOG(ERROR) << "lambda\n"
               << pexpr::CoreExpr{lambda}.ToSExpression() << std::endl;
    LOG(ERROR) << "error-type: " << ret.GetError().class_name()
               << ", error-msg: " << ret.GetError().msg() << std::endl;
  }
  ASSERT_TRUE(ret.HasOkValue());
  const Result<pexpr::PointerType>& opt_pointer_type =
      MethodClass<Val>::TryGet<pexpr::PointerType>(ret.GetOkValue());
  ASSERT_TRUE(opt_pointer_type.HasOkValue());
  const pexpr::PointerType& pointer_type = opt_pointer_type.GetOkValue();
  ASSERT_TRUE(pointer_type.Has<pexpr::CppPointerType<const int32_t*>>());
}

TEST(KernelDefine, FromJson) {
  const std::string json_str = R"(
  [
    "lambda",
    [
      "ctx"
    ],
    [
      "__builtin_let__",
      [
        [
          "__lambda_expr_tmp0",
          [
            "__builtin_getattr__",
            "ctx",
            {
              "str": "module"
            }
          ]
        ],
        [
          "__lambda_expr_tmp1",
          [
            "__builtin_getattr__",
            "ctx",
            {
              "str": "declare_func"
            }
          ]
        ],
        [
          "__lambda_expr_tmp2",
          [
            "__builtin_getattr__",
            "ctx",
            {
              "str": "const_float_ptr"
            }
          ]
        ],
        [
          "__lambda_expr_tmp3",
          [
            "__builtin_getattr__",
            "ctx",
            {
              "str": "const_int32"
            }
          ]
        ],
        [
          "__lambda_expr_tmp4",
          [
            "__builtin_getattr__",
            "ctx",
            {
              "str": "float_ptr"
            }
          ]
        ],
        [
          "__lambda_expr_tmp5",
          [
            "__builtin_list__",
            "__lambda_expr_tmp2",
            "__lambda_expr_tmp3",
            "__lambda_expr_tmp4"
          ]
        ],
        [
          "__lambda_expr_tmp6",
          [
            "__lambda_expr_tmp1",
            {
              "str": "relu"
            },
            "__lambda_expr_tmp5"
          ]
        ],
        [
          "__lambda_expr_tmp7",
          [
            "__builtin_getattr__",
            "ctx",
            {
              "str": "source_code"
            }
          ]
        ],
        [
          "__lambda_expr_tmp8",
          [
            "__lambda_expr_tmp7",
            {
              "str": "\n#include <cstdint>\n#define CINN_WITH_CUDA\n\nextern \"C\" __global__\nvoid relu(const float* input, const int num, float* output) {\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx < num) {\n        output[idx] = input[idx] > 0 ? input[idx] : 0;\n    }\n}\n"
            }
          ]
        ],
        [
          "__lambda_expr_tmp9",
          [
            "__lambda_expr_tmp0",
            "__lambda_expr_tmp6",
            "__lambda_expr_tmp8"
          ]
        ]
      ],
      [
        "__builtin_identity__",
        "__lambda_expr_tmp9"
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
  const auto& core_expr =
      pexpr::ConvertAnfExprToCoreExpr(anf_expr.GetOkValue());
  ASSERT_TRUE(core_expr.Has<pexpr::Atomic<pexpr::CoreExpr>>());
  const auto& atomic = core_expr.Get<pexpr::Atomic<pexpr::CoreExpr>>();
  ASSERT_TRUE(atomic.Has<pexpr::Lambda<pexpr::CoreExpr>>());
  const auto& lambda = atomic.Get<pexpr::Lambda<pexpr::CoreExpr>>();
  pexpr::CpsExprInterpreter<Val> interpreter;
  DefinerCtx<Val> ctx{DefinerRawCtx{}, pexpr::Object<Val>{}};
  const Result<Val>& ret = interpreter.Interpret(lambda, {ctx});
  if (ret.HasError()) {
    LOG(ERROR) << "lambda\n"
               << pexpr::CoreExpr{lambda}.ToSExpression() << std::endl;
    LOG(ERROR) << "error-type: " << ret.GetError().class_name()
               << ", error-msg: " << ret.GetError().msg() << std::endl;
  }
  ASSERT_TRUE(ret.HasOkValue());
  const Result<Module>& opt_module =
      MethodClass<Val>::TryGet<Module>(ret.GetOkValue());
  ASSERT_TRUE(opt_module.HasOkValue());
  const Module& m = opt_module.GetOkValue();
  ASSERT_EQ(m->func_declares->size(), 1);
  const FuncDeclare& func_declare = m->func_declares->at(0);
  const std::string& func_name = func_declare->func_id;
  ASSERT_EQ(func_name, "relu");
  const auto& arg_types = func_declare->arg_types;
  ASSERT_EQ(arg_types->size(), 3);
  ASSERT_TRUE(arg_types->at(0).IsType<const float*>());
  ASSERT_TRUE(arg_types->at(1).IsType<int32_t>());
  ASSERT_TRUE(arg_types->at(2).IsType<float*>());
}

}  // namespace ap::kernel_define::test
