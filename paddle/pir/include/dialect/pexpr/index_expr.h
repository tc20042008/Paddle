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
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace pexpr {

struct IndexTupleExpr;

struct UndefinedIndexExpr : public std::monostate {
  using std::monostate::monostate;
};

struct PtrGetItemImpl {
  std::string ptr_var_name;
  std::shared_ptr<IndexTupleExpr> index_tuple_expr;
  symbol::DimExpr range;

  bool operator==(const PtrGetItemImpl& other) const {
    return (other.ptr_var_name == this->ptr_var_name) &&
           other.index_tuple_expr == this->index_tuple_expr &&
           other.range == this->range;
  }
};

DEFINE_ADT_RC(PtrGetItem, PtrGetItemImpl);

struct IndexExprDomainImpl {
  symbol::DimExpr range;

  bool operator==(const IndexExprDomainImpl& other) const {
    return other.range == this->range;
  }
};

DEFINE_ADT_RC(IndexExprDomain, const IndexExprDomainImpl);

template <typename Expr>
struct IndexExprBroadcastMaskImpl {
  symbol::DimExpr dim;
  Expr index_expr;

  bool operator==(const IndexExprBroadcastMaskImpl& other) const {
    return other.dim == this->dim && other.index_expr == this->index_expr;
  }
};

template <typename Expr>
DEFINE_ADT_RC(IndexExprBroadcastMask, const IndexExprBroadcastMaskImpl<Expr>);

struct SliceImpl {
  symbol::DimExpr start;
  symbol::DimExpr stop;
  symbol::DimExpr step;

  bool operator==(const SliceImpl& other) const {
    return (other.start == this->start) && (other.stop == this->stop) &&
           (other.step == this->step);
  }
};

DEFINE_ADT_RC(Slice, const SliceImpl);

// IndexExprSlice * IndexExprAffine == IdentityFunc if fields are same.
template <typename Expr>
struct IndexExprSliceImpl {
  Slice slice;
  symbol::DimExpr range;
  Expr index_expr;

  bool operator==(const IndexExprSliceImpl& other) const {
    return (other.slice == this->slice) && (other.range == this->range) &&
           (other.index_expr == this->index_expr);
  }
};

template <typename Expr>
DEFINE_ADT_RC(IndexExprSlice, const IndexExprSliceImpl<Expr>);

template <typename Expr>
struct IndexExprAffineImpl {
  Slice slice;
  symbol::DimExpr range;
  Expr index_expr;

  bool operator==(const IndexExprAffineImpl& other) const {
    return (other.slice == this->slice) && (other.range == this->range) &&
           (other.index_expr == this->index_expr);
  }
};

template <typename Expr>
DEFINE_ADT_RC(IndexExprAffine, const IndexExprAffineImpl<Expr>);

template <typename Expr>
using IndexExprBase = std::variant<UndefinedIndexExpr,
                                   PtrGetItem,
                                   IndexExprDomain,
                                   IndexExprBroadcastMask<Expr>,
                                   IndexExprSlice<Expr>,
                                   IndexExprAffine<Expr>,
                                   DisjointUnion<Expr>>;

struct IndexExpr : public IndexExprBase<IndexExpr> {
  using IndexExprBase<IndexExpr>::IndexExprBase;
  DEFINE_ADT_VARIANT_METHODS(IndexExprBase<IndexExpr>);
};

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
  Expr index_tuple_expr;

  bool operator==(const IndexTupleExprPermuteImpl& other) const {
    return other.perms == this->perms &&
           other.index_tuple_expr == this->index_tuple_expr;
  }
};

template <typename Expr>
DEFINE_ADT_RC(IndexTupleExprPermute, const IndexTupleExprPermuteImpl<Expr>);

template <typename Expr>
struct IndexTupleExprReshapeImpl {
  adt::List<symbol::DimExpr> shape;
  Expr index_tuple_expr;

  bool operator==(const IndexTupleExprReshapeImpl& other) const {
    return other.shape == this->shape &&
           other.index_tuple_expr == this->index_tuple_expr;
  }
};
template <typename Expr>
DEFINE_ADT_RC(IndexTupleExprReshape, const IndexTupleExprReshapeImpl<Expr>);

template <typename Expr>
struct IndexTupleExprTransformImpl {
  adt::List<IndexExpr> index_exprs;
  Expr index_tuple_expr;

  bool operator==(const IndexTupleExprTransformImpl& other) const {
    return other.index_exprs == this->index_exprs &&
           other.index_tuple_expr == this->index_tuple_expr;
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
