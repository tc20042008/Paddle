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

#include "paddle/pir/include/dialect/pexpr/index_closure.h"
#include "paddle/pir/include/dialect/pexpr/op_index_tuple_expr_signature.h"
#include "paddle/pir/include/dialect/pexpr/valid_index_expr_builder.h"

namespace pexpr::index_expr {

adt::Result<OpIndexTupleExprSignature> OrderedOneofIndexClosureImpl::operator()(
    const IndexTupleExpr& indexes_expr) const {
  size_t count = 0;
  for (const auto& [_, lambdas] : nice2index_lambdas) {
    for (const auto& lambda : lambdas) {
      const auto& res = CallLambda(lambda, indexes_expr);
      if (res.Has<OpIndexTupleExprSignature>()) {
        return res.Get<OpIndexTupleExprSignature>();
      }
      ++count;
    }
  }
  return ValueError{std::string() + "all index closure failed. tried count: " +
                    std::to_string(count)};
}

adt::Result<OpIndexTupleExprSignature> OrderedOneofIndexClosureImpl::CallLambda(
    const Lambda<CoreExpr>& lambda, const IndexTupleExpr& indexes_expr) const {
  const std::vector<pexpr::index_expr::Val> args{
      closure_data.ctx,
      closure_data.inputs_meta,
      closure_data.outputs_meta,
      closure_data.in_vars,
      Val{IndexExprValue{indexes_expr}}};
  const auto& res = (*this->interpreter)(lambda, args);
  if (res.Has<Error>()) {
    return res.Get<Error>();
  }
  const auto& val = res.Get<Val>();
  if (!val.Has<IndexExprValue>()) {
    return ValueError{
        std::string() +
        "index lambda should return an IndexExprValue object. but `" +
        GetBuiltinTypeName(val) + "` object returned."};
  }
  const auto& index_expr_value = val.Get<IndexExprValue>();
  if (!index_expr_value.Has<OpIndexTupleExprSignature>()) {
    return ValueError{
        std::string() +
        "index lambda should return a OpIndexTupleExprSignature object. but `" +
        GetBuiltinTypeName(val) + "` object returned."};
  }
  return index_expr_value.Get<OpIndexTupleExprSignature>();
}

namespace {

template <typename IndexesTransformApplyT>
adt::Result<OpIndexTupleExprSignature> OpIndexesTransformApply(
    const OpIndexesTransformSignature& indexes_transform_signature,
    const IndexesTransformApplyT& IndexesTransformApply) {
  InIndexTupleExprSignature in_sig;
  for (const auto& transform :
       *indexes_transform_signature.in_signature.descriptors) {
    const auto& converted = IndexesTransformApply(transform);
    ADT_RETURN_IF_ERROR(converted);
    in_sig.descriptors->emplace_back(converted.GetOkValue());
  }
  OutIndexTupleExprSignature out_sig;
  for (const auto& transform :
       *indexes_transform_signature.out_signature.descriptors) {
    const auto& converted = IndexesTransformApply(transform);
    ADT_RETURN_IF_ERROR(converted);
    out_sig.descriptors->emplace_back(converted.GetOkValue());
  }
  return OpIndexTupleExprSignature{in_sig, out_sig};
}

}  // namespace

adt::Result<OpIndexTupleExprSignature> RecordableIndexClosureImpl::operator()(
    const IndexTupleExpr& indexes_expr) const {
  const auto& ApplyTransform = [&](const TrackedIndexesTransform& transform) {
    return transform.Match(
        [&](const adt::IdentityFunc&) -> adt::Result<IndexTupleExpr> {
          return indexes_expr;
        },
        [&](const IndexTupleExpr& tacked_indexes_expr_as_func)
            -> adt::Result<IndexTupleExpr> {
          return ValidIndexExprBuilder().Compose(tacked_indexes_expr_as_func,
                                                 indexes_expr);
        });
  };
  return OpIndexesTransformApply(this->op_indexes_transform_signature,
                                 ApplyTransform);
}

adt::Result<OpIndexTupleExprSignature> IndexClosure::operator()(
    const IndexTupleExpr& indexes_expr) const {
  return Match([&](const auto& impl) { return (*impl)(indexes_expr); });
}

}  // namespace pexpr::index_expr
