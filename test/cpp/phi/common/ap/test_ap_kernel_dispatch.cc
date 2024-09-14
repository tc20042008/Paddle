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

#include "ap/axpr/anf_expr_util.h"
#include "ap/axpr/core_expr.h"
#include "ap/axpr/cps_expr_interpreter.h"
#include "ap/axpr/lambda_expr_builder.h"
#include "ap/kernel/define_ctx_value.h"
#include "ap/kernel/dispatch_ctx_value.h"
#include "ap/kernel/dispatch_ctx_value_method_class.h"
#include "gtest/gtest.h"

namespace ap::kernel_dispatch::test {

TEST(KernelDispatch, CppValue) {
  ap::axpr::LambdaExprBuilder lmbd;
  ap::axpr::AnfExpr anf_expr = lmbd.Lambda({"ctx"}, [&](auto& ctx) {
    ctx.Var("inputs") = ctx.Var("ctx").Attr("inputs");
    ctx.Var("in0") = ctx.Var("inputs").At(0);
    ctx.Var("ptr") = ctx.Var("in0").Attr("data_ptr");
    return ctx.Var("ptr");
  });
  ap::axpr::CoreExpr core_expr = ap::axpr::ConvertAnfExprToCoreExpr(anf_expr);
  ASSERT_TRUE(core_expr.Has<ap::axpr::Atomic<ap::axpr::CoreExpr>>());
  const auto& atomic = core_expr.Get<ap::axpr::Atomic<ap::axpr::CoreExpr>>();
  ASSERT_TRUE(atomic.Has<ap::axpr::Lambda<ap::axpr::CoreExpr>>());
  const auto& lambda = atomic.Get<ap::axpr::Lambda<ap::axpr::CoreExpr>>();
  int32_t number = 30;
  TypedBuffer buffer{
      .buffer = &number, .dtype = phi::DataType::INT32, .size = 1};
  ConstTensor<Val> const_tensor{.tensor_data = ConstTensorData{buffer},
                                .dims = adt::List<Val>{Val{int64_t(1)}}};
  DispatchRawCtx<Val> raw_ctx{
      .inputs = adt::List<Val>{Val{const_tensor}},
      .outputs = adt::List<Val>{},
      .cuda_module = nullptr,
      .func_name2arg_types =
          std::unordered_map<std::string, adt::List<kernel_define::ArgType>>{}};
  ap::axpr::CpsExprInterpreter<Val> interpreter;
  const Result<Val>& ret = interpreter.Interpret(lambda, {raw_ctx});
  if (ret.HasError()) {
    LOG(ERROR) << "lambda\n"
               << ap::axpr::CoreExpr{lambda}.ToSExpression() << std::endl;
    LOG(ERROR) << "error-type: " << ret.GetError().class_name()
               << ", error-msg: " << ret.GetError().msg() << std::endl;
  }
  ASSERT_TRUE(ret.HasOkValue());
  const Result<ap::axpr::PointerValue>& opt_ptr_value =
      ap::axpr::MethodClass<Val>::template TryGet<ap::axpr::PointerValue>(
          ret.GetOkValue());
  ASSERT_TRUE(opt_ptr_value.HasOkValue());
  const ap::axpr::PointerValue& ptr_value = opt_ptr_value.GetOkValue();
  ASSERT_TRUE(ptr_value.Has<const int32_t*>());
  ASSERT_EQ(ptr_value.Get<const int32_t*>(), &number);
}

}  // namespace ap::kernel_dispatch::test
