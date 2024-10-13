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

#include <vector>
#include "ap/axpr/adt.h"
#include "ap/axpr/type.h"
#include "ap/index_expr/index_expr.h"
#include "ap/index_expr/slice.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace ap::index_expr {

struct UndefinedIndexTupleExprImpl : public std::monostate {
  using std::monostate::monostate;

  const char* TypeName() const { return "UndefinedIndexTupleExpr"; }
};
DEFINE_ADT_RC(UndefinedIndexTupleExpr, UndefinedIndexTupleExprImpl);

struct NothingIndexTupleExprImpl : public std::monostate {
  using std::monostate::monostate;

  const char* TypeName() const { return "NothingIndexTupleExpr"; }
};
DEFINE_ADT_RC(NothingIndexTupleExpr, NothingIndexTupleExprImpl);

struct IntArrayLikeIndexTupleExprImpl : public std::monostate {
  using std::monostate::monostate;

  const char* TypeName() const { return "IntArrayLikeIndexTupleExpr"; }
};
DEFINE_ADT_RC(IntArrayLikeIndexTupleExpr, IntArrayLikeIndexTupleExprImpl);

struct IndexTupleExprDomainImpl {
  adt::List<symbol::DimExpr> ranges;
  bool operator==(const IndexTupleExprDomainImpl& other) const {
    return other.ranges == this->ranges;
  }

  const char* TypeName() const { return "IndexTupleExprDomain"; }
};
DEFINE_ADT_RC(IndexTupleExprDomain, const IndexTupleExprDomainImpl);

template <typename Expr>
struct IndexTupleExprPermuteImpl {
  adt::List<int64_t> perms;
  Expr indexes_expr;

  bool operator==(const IndexTupleExprPermuteImpl& other) const {
    return other.perms == this->perms &&
           other.indexes_expr == this->indexes_expr;
  }

  const char* TypeName() const { return "IndexTupleExprPermute"; }
};

template <typename Expr>
DEFINE_ADT_RC(IndexTupleExprPermute, const IndexTupleExprPermuteImpl<Expr>);

template <typename Expr>
struct IndexTupleExprReshapeImpl {
  adt::List<symbol::DimExpr> shape;
  Expr indexes_expr;

  bool operator==(const IndexTupleExprReshapeImpl& other) const {
    return other.shape == this->shape &&
           other.indexes_expr == this->indexes_expr;
  }

  const char* TypeName() const { return "IndexTupleExprReshape"; }
};
template <typename Expr>
DEFINE_ADT_RC(IndexTupleExprReshape, const IndexTupleExprReshapeImpl<Expr>);

template <typename Expr>
struct IndexTupleExprTransformImpl {
  adt::List<IndexExpr> index_exprs;
  Expr indexes_expr;

  bool operator==(const IndexTupleExprTransformImpl& other) const {
    return other.index_exprs == this->index_exprs &&
           other.indexes_expr == this->indexes_expr;
  }

  const char* TypeName() const { return "IndexTupleExprTransform"; }
};
template <typename Expr>
DEFINE_ADT_RC(IndexTupleExprTransform, const IndexTupleExprTransformImpl<Expr>);

template <typename Expr>
using IndexTupleExprBase = std::variant<UndefinedIndexTupleExpr,
                                        NothingIndexTupleExpr,
                                        IntArrayLikeIndexTupleExpr,
                                        IndexTupleExprDomain,
                                        IndexTupleExprPermute<Expr>,
                                        IndexTupleExprReshape<Expr>,
                                        IndexTupleExprTransform<Expr>>;

struct IndexTupleExpr : public IndexTupleExprBase<IndexTupleExpr> {
  using IndexTupleExprBase<IndexTupleExpr>::IndexTupleExprBase;
  DEFINE_ADT_VARIANT_METHODS(IndexTupleExprBase<IndexTupleExpr>);

  const char* TypeName() const {
    return Match([](const auto& impl) { return impl->TypeName(); });
  }
};

}  // namespace ap::index_expr

namespace ap::axpr {

template <>
struct TypeImpl<index_expr::IndexTupleExpr> : public std::monostate {
  using value_type = index_expr::IndexTupleExpr;

  const char* Name() const { return "IndexTupleExpr"; }
};

}  // namespace ap::axpr
