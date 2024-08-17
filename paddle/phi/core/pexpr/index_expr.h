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
#include "paddle/phi/core/pexpr/adt.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace pexpr {

struct IndexTupleExpr;

struct PtrGetItemImpl {
  std::string ptr_var_name;
  std::shared_ptr<IndexTupleExpr> index_tuple_expr;

  bool operator==(const PtrGetItemImpl& other) const {
    return (other.ptr_var_name == this->ptr_var_name) &&
           other.index_tuple_expr == this->index_tuple_expr;
  }
};

DEFINE_ADT_RC(PtrGetItem, PtrGetItemImpl);

struct IndexDomainImpl {
  symbol::DimExpr domain;

  bool operator==(const IndexDomainImpl& other) const {
    return other.domain == this->domain;
  }
};
DEFINE_ADT_RC(IndexDomain, const IndexDomainImpl);

template <typename Expr>
struct IndexBroadcastMaskImpl {
  symbol::DimExpr dim;
  Expr index_expr;

  bool operator==(const IndexBroadcastMaskImpl& other) const {
    return other.dim == this->dim && other.index_expr == this->index_expr;
  }
};

template <typename Expr>
DEFINE_ADT_RC(IndexTupleBroadcastMask, const IndexBroadcastMaskImpl<Expr>);

// IndexSlice * IndexAffine == IdentityFunc if fields are same.
template <typename Expr>
struct IndexSliceImpl {
  symbol::DimExpr start;
  symbol::DimExpr stop;
  int64_t step;
  Expr index_expr;

  bool operator==(const IndexSliceImpl& other) const {
    return (other.start == this->start) && (other.stop == this->stop) &&
           (other.step == this->step) && (other.index_expr == this->index_expr);
  }
};

template <typename Expr>
DEFINE_ADT_RC(IndexSlice, const IndexSliceImpl<Expr>);

template <typename Expr>
struct IndexAffineImpl {
  symbol::DimExpr start;
  symbol::DimExpr stop;
  int64_t step;
  Expr index_expr;

  bool operator==(const IndexAffineImpl& other) const {
    return (other.start == this->start) && (other.stop == this->stop) &&
           (other.step == this->step) && (other.index_expr == this->index_expr);
  }
};

template <typename Expr>
DEFINE_ADT_RC(IndexAffine, const IndexAffineImpl<Expr>);

template <typename Expr>
using IndexExprBase = std::variant<Undefined,
                                   PtrGetItem,
                                   IndexDomain<Expr>,
                                   IndexBroadcastMask<Expr>,
                                   IndexSlice<Expr>,
                                   IndexAffine<Expr>,
                                   DisjointUnion<Expr>>;

struct IndexExpr : public IndexExprBase<IndexExpr> {
  using IndexExprBase<IndexExpr>::IndexExprBase;
  DEFINE_ADT_VARIANT_METHODS(IndexExprBase<IndexExpr>);
};

struct IntArrayLikeIndexes : public std::monostate {
  using std::monostate::monostate;
};

struct IndexTupleDomainImpl {
  std::vector<symbol::DimExpr> domains;

  bool operator==(const IndexTupleDomainImpl& other) const {
    return other.domains == this->domains;
  }
};
DEFINE_ADT_RC(IndexTupleDomain, const IndexTupleDomainImpl);

template <typename Expr>
struct IndexTuplePermuteImpl {
  std::vector<int64_t> perms;
  Expr index_tuple_expr;

  bool operator==(const IndexTuplePermuteImpl& other) const {
    return other.perms == this->perms &&
           other.index_tuple_expr == this->index_tuple_expr;
  }
};

template <typename Expr>
DEFINE_ADT_RC(IndexTuplePermute, const IndexTuplePermuteImpl<Expr>);

template <typename Expr>
struct IndexTupleReshapeImpl {
  std::vector<symbol::DimExpr> shape;
  Expr index_tuple_expr;

  bool operator==(const IndexTupleReshapeImpl& other) const {
    return other.shape == this->shape &&
           other.index_tuple_expr == this->index_tuple_expr;
  }
};
template <typename Expr>
DEFINE_ADT_RC(IndexTupleReshape, const IndexTupleReshapeImpl<Expr>);

template <typename Expr>
struct IndexTupleMatchCallImpl {
  std::vector<IndexExpr> index_exprs;
  Expr index_tuple_expr;

  bool operator==(const IndexTupleMatchCallImpl& other) const {
    return other.index_exprs == this->index_exprs &&
           other.index_tuple_expr == this->index_tuple_expr;
  }
};
template <typename Expr>
DEFINE_ADT_RC(IndexTupleMatchCall, const IndexTupleMatchCallImpl<Expr>);

template <typename Expr>
using IndexTupleExprBase = std::variant<Undefined,
                                        Nothing,
                                        IntArrayLikeIndexes,
                                        IndexTupleDomain,
                                        IndexTuplePermute<Expr>,
                                        IndexTupleReshape<Expr>,
                                        IndexTupleMatchCall<Expr>>;

struct IndexTupleExpr : public IndexTupleExprBase<IndexTupleExpr> {
  using IndexTupleExprBase<IndexTupleExpr>::IndexTupleExprBase;
  DEFINE_ADT_VARIANT_METHODS(IndexTupleExprBase<IndexTupleExpr>);
};

}  // namespace pexpr
