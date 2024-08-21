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
#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/dialect/pexpr/builtin_functions.h"
#include "paddle/pir/include/dialect/pexpr/core_expr.h"
#include "paddle/pir/include/dialect/pexpr/index_expr.h"
#include "paddle/pir/include/dialect/pexpr/value.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"
#include "paddle/pir/include/dialect/shape/utils/shape_or_data_expr.h"

namespace pexpr::index_expr {

struct InIndexTupleExprSignature {
  adt::List<IndexTupleExpr> in_indexes_exprs;

  bool operator==(const InIndexTupleExprSignature& other) const {
    return other.in_indexes_exprs == this->in_indexes_exprs;
  }
};

struct OutIndexTupleExprSignature {
  adt::List<IndexTupleExpr> out_indexes_exprs;

  bool operator==(const OutIndexTupleExprSignature& other) const {
    return other.out_indexes_exprs == this->out_indexes_exprs;
  }
};

struct OpIndexTupleExprSignature {
  InIndexTupleExprSignature in_signature;
  OutIndexTupleExprSignature out_signature;

  bool operator==(const OpIndexTupleExprSignature& other) const {
    return other.in_signature == this->in_signature &&
           other.out_signature == this->out_signature;
  }
};

template <typename Val>
using IndexExprValueBase = std::variant<symbol::DimExpr,
                                        Slice,
                                        IndexExpr,
                                        IndexTupleExpr,
                                        InIndexTupleExprSignature,
                                        OutIndexTupleExprSignature,
                                        OpIndexTupleExprSignature>;

struct IndexExprValue : public IndexExprValueBase<Value<IndexExprValue>> {
  using IndexExprValueBase<Value<IndexExprValue>>::IndexExprValueBase;
  DEFINE_ADT_VARIANT_METHODS(IndexExprValueBase<Value<IndexExprValue>>);
};

using Val = Value<IndexExprValue>;

using Env = Environment<Val>;
using EnvMgr = EnvironmentManager<Val>;

inline Result<Val> CustomGetAttr(const IndexExprValue&,
                                 const std::string& name) {
  return AttributeError{std::string("no attribute '") + name + "' found."};
}

}  // namespace pexpr::index_expr
