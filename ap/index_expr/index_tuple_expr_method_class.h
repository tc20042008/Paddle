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

#include "ap/axpr/method_class.h"
#include "ap/index_expr/index_expr_builtin_functions.h"
#include "ap/index_expr/index_tuple_expr.h"

namespace ap::index_expr {

template <typename ValueT>
struct IndexTupleExprMethodClass {
  using This = IndexTupleExprMethodClass;
  using Self = IndexTupleExpr;

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, axpr::TryGetImpl<std::string>(attr_name_val));
    if (attr_name == "reshape") {
      return axpr::Method<ValueT>{self, &MakeIndexTupleExprReshape<ValueT>};
    }
    if (attr_name == "permute") {
      return axpr::Method<ValueT>{self, &MakeIndexTupleExprPermute<ValueT>};
    }
    if (attr_name == "transform") {
      return axpr::Method<ValueT>{self, &MakeIndexTupleExprTransform<ValueT>};
    }
    return adt::errors::TypeError{std::string() +
                                  "'IndexTupleExpr' object has no attribute '" +
                                  attr_name + "'."};
  }
};

}  // namespace ap::index_expr

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, index_expr::IndexTupleExpr>
    : public index_expr::IndexTupleExprMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<index_expr::IndexTupleExpr>>
    : public EmptyMethodClass<ValueT> {};

}  // namespace ap::axpr
