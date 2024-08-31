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
#include "paddle/phi/common/ap/kernel_definer_interpreter.h"
#include "paddle/pir/include/dialect/pexpr/anf_expr_util.h"
#include "paddle/pir/include/dialect/pexpr/core_expr.h"
#include "paddle/pir/include/dialect/pexpr/lambda_expr_builder.h"

namespace ap::kernel_define::test {

TEST(KernelDefine, ArgType) {
  pexpr::LambdaExprBuilder lmbd;
  pexpr::AnfExpr anf_expr = lmbd.Lambda({"ctx"}, [&](auto& ctx) {
    return ctx.Var("ctx").Attr("const_int32_ptr").Call();
  });
  pexpr::CoreExpr core_expr = pexpr::ConvertAnfExprToCoreExpr(anf_expr);
  ASSERT_TRUE(core_expr.Has<pexpr::Atomic<pexpr::CoreExpr>>());
  const auto& atomic = core_expr.Get<pexpr::Atomic<pexpr::CoreExpr>>();
  ASSERT_TRUE(atomic.Has<pexpr::Lambda<pexpr::CoreExpr>>());
  const auto& lambda = atomic.Get<pexpr::Lambda<pexpr::CoreExpr>>();
  KernelDefinerInterpreter interpreter;
  DefinerCtx<Val> ctx{DefinerRawCtx{}, pexpr::Object<Val>{}};
  const Result<Val>& ret = interpreter(lambda, ctx);
  if (ret.HasError()) {
    LOG(ERROR) << "lambda\n"
               << pexpr::CoreExpr{lambda}.ToSExpression() << std::endl;
    LOG(ERROR) << "error-type: " << ret.GetError().class_name()
               << ", error-msg: " << ret.GetError().msg() << std::endl;
  }
  ASSERT_TRUE(ret.HasOkValue());
  const Result<ArgType>& opt_arg_type =
      CastToCustomValue<ArgType>(ret.GetOkValue());
  ASSERT_TRUE(opt_arg_type.HasOkValue());
  const ArgType& arg_type = opt_arg_type.GetOkValue();
  ASSERT_TRUE(arg_type.Has<CppArgType<const int32_t*>>());
}

}  // namespace ap::kernel_define::test
