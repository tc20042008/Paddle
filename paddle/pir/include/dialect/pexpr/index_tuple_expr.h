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
#include "paddle/pir/include/dialect/pexpr/adt.h"
#include "paddle/pir/include/dialect/pexpr/index_expr.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace pexpr {

struct UndefinedIndexTupleExpr : public std::monostate {
  using std::monostate::monostate;
};

struct NothingIndexTupleExpr : public std::monostate {
  using std::monostate::monostate;
};

struct IntArrayLikeIndexTupleExpr : public std::monostate {
  using std::monostate::monostate;
};

struct IndexTupleExprDomainImpl {
  adt::List<symbol::DimExpr> ranges;
  bool operator==(const IndexTupleExprDomainImpl& other) const {
    return other.ranges == this->ranges;
  }
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
};

}  // namespace pexpr
