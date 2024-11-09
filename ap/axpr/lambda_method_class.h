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

#pragma once

#include "ap/axpr/anf_expr_helper.h"
#include "ap/axpr/anf_expr_util.h"
#include "ap/axpr/lambda.h"

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, Lambda<CoreExpr>>
    : public EmptyMethodClass<ValueT> {
  adt::Result<ValueT> ToString(const Lambda<CoreExpr>& lambda) {
    const auto& anf_expr = ConvertCoreExprToAnfExpr(lambda);
    ADT_LET_CONST_REF(anf_atomic, anf_expr.template TryGet<Atomic<AnfExpr>>());
    ADT_LET_CONST_REF(anf_lambda,
                      anf_atomic.template TryGet<Lambda<AnfExpr>>());
    AnfExprHelper anf_expr_helper;
    ADT_LET_CONST_REF(anf_expr_str,
                      anf_expr_helper.FunctionToString(anf_lambda));
    return anf_expr_str;
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<Lambda<CoreExpr>>>
    : public EmptyMethodClass<ValueT> {};

}  // namespace ap::axpr
