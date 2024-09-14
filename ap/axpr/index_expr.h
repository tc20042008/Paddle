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
#include "ap/axpr/slice.h"
#include "ap/axpr/type.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace ap::axpr {

struct IndexTupleExpr;

struct UndefinedIndexExpr : public std::monostate {
  using std::monostate::monostate;
};

struct PtrGetItemImpl {
  std::string ptr_var_name;
  std::shared_ptr<IndexTupleExpr> indexes_expr;
  symbol::DimExpr range;

  bool operator==(const PtrGetItemImpl& other) const {
    return (other.ptr_var_name == this->ptr_var_name) &&
           other.indexes_expr == this->indexes_expr &&
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

template <>
struct TypeImpl<IndexExpr> : public std::monostate {
  using value_type = IndexExpr;

  const char* Name() const { return "IndexExpr"; }
};

}  // namespace ap::axpr
