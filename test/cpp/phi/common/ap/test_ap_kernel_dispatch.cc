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

#include "gtest/gtest.h"
#include "paddle/phi/common/ap/define_ctx_value.h"
#include "paddle/phi/common/ap/dispatch_ctx_value.h"
#include "paddle/phi/common/ap/kernel_dispatcher_interpreter.h"
#include "paddle/pir/include/dialect/pexpr/anf_expr_util.h"
#include "paddle/pir/include/dialect/pexpr/core_expr.h"
#include "paddle/pir/include/dialect/pexpr/lambda_expr_builder.h"

namespace ap::kernel_dispatch::test {

TEST(KernelDispatch, CppValue) {
  pexpr::LambdaExprBuilder lmbd;
  pexpr::AnfExpr anf_expr = lmbd.Lambda({"ctx"}, [&](auto& ctx) {
    ctx.Var("inputs") = ctx.Var("ctx").Attr("inputs");
    ctx.Var("in0") = ctx.Var("inputs").At(0);
    ctx.Var("ptr") = ctx.Var("in0").Attr("data_ptr");
    return ctx.Var("ptr");
  });
  pexpr::CoreExpr core_expr = pexpr::ConvertAnfExprToCoreExpr(anf_expr);
  ASSERT_TRUE(core_expr.Has<pexpr::Atomic<pexpr::CoreExpr>>());
  const auto& atomic = core_expr.Get<pexpr::Atomic<pexpr::CoreExpr>>();
  ASSERT_TRUE(atomic.Has<pexpr::Lambda<pexpr::CoreExpr>>());
  const auto& lambda = atomic.Get<pexpr::Lambda<pexpr::CoreExpr>>();
  int32_t number = 30;
  TypedBuffer buffer{
      .buffer = &number, .dtype = phi::DataType::INT32, .size = 1};
  ConstTensor<Val> const_tensor{.tensor_data = ConstTensorData{buffer},
                                .dims = adt::List<Val>{Val{int64_t(1)}}};
  DispatchRawContext<Val> raw_ctx{
      .inputs = adt::List<Val>{Val{const_tensor}},
      .outputs = adt::List<Val>{},
      .cuda_module = nullptr,
      .func_name2arg_types =
          std::unordered_map<std::string, adt::List<kernel_define::ArgType>>{}};
  KernelDispatcherInterpreter interpreter;
  const Result<Val>& ret = interpreter(lambda, raw_ctx);
  if (ret.HasError()) {
    LOG(ERROR) << "lambda\n"
               << pexpr::CoreExpr{lambda}.ToSExpression() << std::endl;
    LOG(ERROR) << "error-type: " << ret.GetError().class_name()
               << ", error-msg: " << ret.GetError().msg() << std::endl;
  }
  ASSERT_TRUE(ret.HasOkValue());
  const Result<CppValue>& opt_cpp_value =
      CastToCustomValue<CppValue>(ret.GetOkValue());
  ASSERT_TRUE(opt_cpp_value.HasOkValue());
  const CppValue& cpp_value = opt_cpp_value.GetOkValue();
  ASSERT_TRUE(cpp_value.Has<const int32_t*>());
  ASSERT_EQ(cpp_value.Get<const int32_t*>(), &number);
}

}  // namespace ap::kernel_dispatch::test
