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

#include "paddle/pir/include/dialect/pexpr/error.h"
#include "paddle/pir/include/dialect/pexpr/index_expr.h"

namespace pexpr {

class ValidIndexExprBuilder {
 public:
  ValidIndexExprBuilder() {}
  ValidIndexExprBuilder(const ValidIndexExprBuilder&) = delete;
  ValidIndexExprBuilder(ValidIndexExprBuilder&&) = delete;

  Result<IndexExpr> BroadcastMask(const symbol::DimExpr& dim_expr,
                                  const IndexExpr& index_expr);
  Result<IndexExpr> Slice(const Slice& slice,
                          const symbol::DimExpr& range,
                          const IndexExpr& index_expr);
  Result<IndexExpr> Affine(const Slice& slice,
                           const symbol::DimExpr& range,
                           const IndexExpr& index_expr);
  Result<IndexExpr> DisjointUnion(const IndexExpr& lhs, const IndexExpr& rhs);

  Result<IndexTupleExpr> Permute(const adt::List<int64_t>& perms,
                                 const IndexTupleExpr& indexes_expr);
  Result<IndexTupleExpr> Reshape(const adt::List<symbol::DimExpr>& shape,
                                 const IndexTupleExpr& indexes_expr);
  Result<IndexTupleExpr> Transform(
      const adt::List<IndexExpr>& transform_index_exprs,
      const IndexTupleExpr& indexes_expr);

  // outter(inner(x)) == (outter . inner)(x)
  Result<IndexTupleExpr> Compose(const IndexTupleExpr& outter,
                                 const IndexTupleExpr& inner);
};

}  // namespace pexpr
