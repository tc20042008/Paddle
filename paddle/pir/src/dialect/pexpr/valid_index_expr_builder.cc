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

#include "paddle/pir/include/dialect/pexpr/valid_index_expr_builder.h"
#include "paddle/pir/include/dialect/pexpr/index_expr.h"
#include "paddle/pir/include/dialect/pexpr/index_expr_util.h"
#include "paddle/pir/include/dialect/pexpr/index_tuple_expr.h"

namespace pexpr::index_expr {

Result<IndexExpr> ValidIndexExprBuilder::BroadcastMask(
    const symbol::DimExpr& dim_expr, const IndexExpr& index_expr) {
  return IndexExprBroadcastMask<IndexExpr>{dim_expr, index_expr};
}

Result<IndexExpr> ValidIndexExprBuilder::Slice(const pexpr::Slice& slice,
                                               const symbol::DimExpr& range,
                                               const IndexExpr& index_expr) {
  return IndexExprSlice<IndexExpr>{slice, range, index_expr};
}

Result<IndexExpr> ValidIndexExprBuilder::Affine(const pexpr::Slice& slice,
                                                const symbol::DimExpr& range,
                                                const IndexExpr& index_expr) {
  return IndexExprAffine<IndexExpr>{slice, range, index_expr};
}

Result<IndexExpr> ValidIndexExprBuilder::DisjointUnion(
    const IndexExpr& lhs_index_expr, const IndexExpr& rhs_index_expr) {
  const auto& lhs_domain = IndexExprGetDomain(lhs_index_expr);
  const auto& rhs_domain = IndexExprGetDomain(rhs_index_expr);
  const auto& pattern_match = ::common::Overloaded{
      [](const symbol::DimExpr& lhs, const symbol::DimExpr& rhs) {
        return lhs == rhs;
      },
      [](const auto&, const auto&) { return false; }};
  const bool do_equal =
      std::visit(pattern_match, lhs_domain.variant(), rhs_domain.variant());
  if (!do_equal) {
    return TypeError{
        "domain of `lhs_index_expr' does not equal to domain of "
        "`rhs_index_expr'"};
  }
  return pexpr::DisjointUnion<IndexExpr>{lhs_index_expr, rhs_index_expr};
}

namespace {

template <typename PermsT>
bool IsValidPerm(const PermsT& perms) {
  std::vector<bool> idx2touched(perms->size(), false);
  for (int64_t perm : *perms) {
    if (perm < 0) {
      return false;
    }
    if (perm >= perms->size()) {
      return false;
    }
    idx2touched[perm] = true;
  }
  for (bool touched : idx2touched) {
    if (!touched) {
      return false;
    }
  }
  return true;
}

}  // namespace

Result<IndexTupleExpr> ValidIndexExprBuilder::Permute(
    const adt::List<int64_t>& perms, const IndexTupleExpr& indexes_expr) {
  if (!IsValidPerm(perms)) {
    return InvalidArgumentError{"argument `perms` is not valid"};
  }
  const auto& rank = IndexTupleExprGetRank(indexes_expr);
  if (!rank.Has<int64_t>()) {
    return InvalidArgumentError{
        "wrong indexes_expr argument for IndexTupleExprPermute"};
  }
  if (rank.Get<int64_t>() != perms->size()) {
    return InvalidArgumentError{std::string(
        "the rank of perms does not equal to the rank of "
        "indexes_expr. rank(perm): " +
        std::to_string(perms->size()) +
        ", rank(indexes_expr): " + std::to_string(rank.Get<int64_t>()))};
  }
  return IndexTupleExprPermute<IndexTupleExpr>{perms, indexes_expr};
}

namespace {

template <typename ShapeT>
bool ContainsNegative(const ShapeT& shape) {
  for (const auto& dim : *shape) {
    if (!dim.template Has<int64_t>()) {
      continue;
    }
    if (dim.template Get<int64_t>() < 0) {
      return true;
    }
  }
  return false;
}

template <typename DimExprsT>
symbol::DimExpr Product(const DimExprsT& dim_exprs) {
  symbol::DimExpr ret_expr{1};
  for (const auto& dim : *dim_exprs) {
    ret_expr = ret_expr * dim;
  }
  return ret_expr;
}

bool ProductEqual(const auto& lhs, const auto& rhs) {
  return Product(lhs) == Product(rhs);
}

}  // namespace

Result<IndexTupleExpr> ValidIndexExprBuilder::Reshape(
    const adt::List<symbol::DimExpr>& shape,
    const IndexTupleExpr& indexes_expr) {
  if (ContainsNegative(shape)) {
    return InvalidArgumentError{
        "dims in argument `shape` have negative integer"};
  }
  const auto& opt_ranges = IndexTupleExprGetRanges(indexes_expr);
  if (opt_ranges.Has<Nothing>()) {
    return InvalidArgumentError{
        "argument `indexes_expr` is not a ranked IndexTupleExpr"};
  }
  if (!ProductEqual(shape, opt_ranges.Get<adt::List<symbol::DimExpr>>())) {
    return InvalidArgumentError{
        "product of argument `shape` does not equal to elements of "
        "`indexes_expr`"};
  }
  return IndexTupleExprReshape<IndexTupleExpr>{shape, indexes_expr};
}

Result<IndexTupleExpr> ValidIndexExprBuilder::Transform(
    const adt::List<IndexExpr>& transform_index_exprs,
    const IndexTupleExpr& indexes_expr) {
  const auto& opt_rank = IndexTupleExprGetRank(indexes_expr);
  if (!opt_rank.Has<int64_t>()) {
    return TypeError{
        "The first argument of IndexTupleExprTransform must be a ranked "
        "IndexTupleExpr."};
  }
  const auto& opt_ranges = IndexTupleExprGetRanges(indexes_expr);
  if (!opt_ranges.Has<adt::List<symbol::DimExpr>>()) {
    return RuntimeError{"error occured where calling IndexTupleExprGetDims"};
  }
  const auto& ranges = opt_ranges.Get<adt::List<symbol::DimExpr>>();
  if (opt_rank.Get<int64_t>() != transform_index_exprs->size()) {
    return TypeError{
        "The rank of first argument must equal to number of lambdas."};
  }
  adt::List<symbol::DimExpr> domains{};
  domains->reserve(transform_index_exprs->size());
  for (const auto& index_expr : *transform_index_exprs) {
    const auto& domain = IndexExprGetDomain(index_expr);
    if (!domain.Has<symbol::DimExpr>()) {
      return TypeError{"one of transform_index_exprs has no demain."};
    }
    domains->emplace_back(domain.Get<symbol::DimExpr>());
  }
  if (ranges != domains) {
    return TypeError{
        "domain of `transform_index_exprs' does not equal to range of "
        "`indexes_expr'."};
  }
  return IndexTupleExprTransform<IndexTupleExpr>{transform_index_exprs,
                                                 indexes_expr};
}

Result<IndexTupleExpr> ValidIndexExprBuilder::Compose(
    const IndexTupleExpr& outter, const IndexTupleExpr& inner) {
  return outter.Match(
      [&](const UndefinedIndexTupleExpr& impl) -> Result<IndexTupleExpr> {
        return impl;
      },
      [&](const NothingIndexTupleExpr& impl) -> Result<IndexTupleExpr> {
        return impl;
      },
      [&](const IntArrayLikeIndexTupleExpr& impl) -> Result<IndexTupleExpr> {
        return impl;
      },
      [&](const IndexTupleExprDomain& domain) -> Result<IndexTupleExpr> {
        const auto& ranges = IndexTupleExprGetRanges(inner);
        if (ranges.Has<Nothing>()) {
          return TypeError{"`inner_indexes_expr' has no range."};
        }
        if (ranges.Get<adt::List<symbol::DimExpr>>() != domain->ranges) {
          return TypeError{
              "the domain of `outter_indexes_expr' does not equal to the range "
              "of `inner_indexes_expr'."};
        }
        return inner;
      },
      [&](const IndexTupleExprPermute<IndexTupleExpr>& perm)
          -> Result<IndexTupleExpr> {
        const auto& composed_inner = Compose(perm->indexes_expr, inner);
        if (composed_inner.Has<Error>()) {
          return composed_inner.Get<Error>();
        }
        return Permute(perm->perms, composed_inner.Get<IndexTupleExpr>());
      },
      [&](const IndexTupleExprReshape<IndexTupleExpr>& reshape)
          -> Result<IndexTupleExpr> {
        const auto& composed_inner = Compose(reshape->indexes_expr, inner);
        if (composed_inner.Has<Error>()) {
          return composed_inner.Get<Error>();
        }
        return Reshape(reshape->shape, composed_inner.Get<IndexTupleExpr>());
      },
      [&](const IndexTupleExprTransform<IndexTupleExpr>& transform)
          -> Result<IndexTupleExpr> {
        const auto& composed_inner = Compose(transform->indexes_expr, inner);
        if (composed_inner.Has<Error>()) {
          return composed_inner.Get<Error>();
        }
        return Transform(transform->index_exprs,
                         composed_inner.Get<IndexTupleExpr>());
      });
}

}  // namespace pexpr::index_expr
