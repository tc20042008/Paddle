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

  adt::Result<ValueT> ToString(const Self& self) { return self.ToString(); }

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

template <typename ValueT>
struct TypeImplIndexTupleExprMethodClass {
  using This = TypeImplIndexTupleExprMethodClass;
  using Self = axpr::TypeImpl<IndexTupleExpr>;

  adt::Result<ValueT> GetAttr(const Self&, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    if (attr_name == "Domain") {
      return &This::StaticConstructIndexTupleExprDomain;
    }
    return adt::errors::AttributeError{
        std::string() + "'IndexTupleExpr' has no static attribute '" +
        attr_name + "'."};
  }

  static adt::Result<ValueT> StaticConstructIndexTupleExprDomain(
      const ValueT&, const std::vector<ValueT>& args) {
    return This{}.ConstructIndexTupleExprDomain(args);
  }

  adt::Result<ValueT> ConstructIndexTupleExprDomain(
      const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "'IndexTupleExpr.Domain' takes 1 argument but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(list, args.at(0).template TryGet<adt::List<ValueT>>())
        << adt::errors::TypeError{std::string() +
                                  "the argument 1 of 'IndexTupleExpr.Domain' "
                                  "should a list of DimExpr."};
    adt::List<symbol::DimExpr> dim_exprs;
    dim_exprs->reserve(list->size());
    for (const auto& arg : *list) {
      ADT_LET_CONST_REF(dim_expr, CastToDimExpr(arg))
          << adt::errors::TypeError{std::string() +
                                    "the argument 1 of 'IndexTupleExpr.Domain' "
                                    "should a list of DimExpr."};
      dim_exprs->emplace_back(dim_expr);
    }
    return IndexTupleExpr{IndexTupleExprDomain{dim_exprs}};
  }

  adt::Result<symbol::DimExpr> CastToDimExpr(const ValueT& val) {
    const auto& opt_dim_expr = TryGetDimExpr(val);
    return opt_dim_expr.Match(
        [](const symbol::DimExpr& dim_expr) -> adt::Result<symbol::DimExpr> {
          return dim_expr;
        },
        [](const adt::Nothing&) -> adt::Result<symbol::DimExpr> {
          return adt::errors::ValueError{"CastToDimExpr failed."};
        });
  }
};

}  // namespace ap::index_expr

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, index_expr::IndexTupleExpr>
    : public index_expr::IndexTupleExprMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<index_expr::IndexTupleExpr>>
    : public index_expr::TypeImplIndexTupleExprMethodClass<ValueT> {};

}  // namespace ap::axpr
