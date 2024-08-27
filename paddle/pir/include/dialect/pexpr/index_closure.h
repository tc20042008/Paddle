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

#include <variant>
#include "paddle/pir/include/dialect/pexpr/core_expr.h"
#include "paddle/pir/include/dialect/pexpr/error.h"
#include "paddle/pir/include/dialect/pexpr/index_expr.h"
#include "paddle/pir/include/dialect/pexpr/index_expr_interpreter.h"
#include "paddle/pir/include/dialect/pexpr/op_index_tuple_expr_signature.h"

namespace pexpr::index_expr {

using NativeIndexClosure = std::function<adt::Result<OpIndexTupleExprSignature>(
    const IndexTupleExpr&)>;

struct IndexClosureData {
  const pexpr::Val ctx;
  const std::List<pexpr::Val> inputs_meta;
  const std::List<pexpr::Val> outputs_meta;
  const std::List<pexpr::Val> in_vars;
};

using Nice2IndexLambdas = std::map<int64_t, std::vector<Lambda<CoreExpr>>>;

struct OrderedOneofIndexClosureImpl {
  std::shared_ptr<IndexExprInterpreter> interpreter;
  IndexClosureData closure_data;
  Nice2IndexLambdas nice2index_lambdas;

  adt::Result<OpIndexTupleExprSignature> operator()(
      const IndexTupleExpr&) const;

 private:
  adt::Result<OpIndexTupleExprSignature> CallLambda(
      const Lambda<CoreExpr>& lambda, const IndexTupleExpr&) const;
};
DEFINE_ADT_RC(OrderedOneofIndexClosure, OrderedOneofIndexClosureImpl);

using TrackedIndexesTransformImpl =
    std::variant<adt::IdentityFunc, pexpr::IndexTupleExpr>;

struct TrackedIndexesTransform : public TrackedIndexesTransformImpl {
  using TrackedIndexesTransformImpl::TrackedIndexesTransformImpl;
  DEFINE_ADT_VARIANT_METHODS(TrackedIndexesTransformImpl);
};

using OpIndexesTransformSignature =
    pexpr::index_expr::OpSignature<TrackedIndexesTransform>;

struct RecordableIndexClosure {
  OpIndexesTransformSignature op_indexes_transform_signature;

  adt::Result<OpIndexTupleExprSignature> operator()(
      const IndexTupleExpr&) const;

  bool operator==(const RecordableIndexClosure& other) const {
    return other.op_indexes_transform_signature ==
           this->op_indexes_transform_signature;
  }
};

using IndexClosureImpl = std::variant<NativeIndexClosure,
                                      OrderedOneofIndexClosure,
                                      RecordableIndexClosure>;

struct IndexClosure : public IndexClosureImpl {
  using IndexClosureImpl::IndexClosureImpl;
  DEFINE_ADT_RC(IndexClosureImpl);

  adt::Result<OpIndexTupleExprSignature> operator()(
      const IndexTupleExpr& indexes_expr) const;
};

}  // namespace pexpr::index_expr
