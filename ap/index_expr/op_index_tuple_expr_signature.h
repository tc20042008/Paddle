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
#include "ap/axpr/type.h"
#include "ap/index_expr/index_expr.h"
#include "ap/index_expr/index_tuple_expr.h"
#include "ap/index_expr/op_signature.h"

namespace ap::index_expr {

using InIndexTupleExprSignature = InputSignature<IndexTupleExpr>;
using OutIndexTupleExprSignature = OutputSignature<IndexTupleExpr>;
using OpIndexTupleExprSignature = OpSignature<IndexTupleExpr>;

}  // namespace ap::index_expr

namespace ap::axpr {

template <>
struct TypeImpl<index_expr::InIndexTupleExprSignature> : public std::monostate {
  using value_type = index_expr::InIndexTupleExprSignature;

  const char* Name() const { return "InIndexTupleExprSignature"; }
};

template <>
struct TypeImpl<index_expr::OutIndexTupleExprSignature>
    : public std::monostate {
  using value_type = index_expr::OutIndexTupleExprSignature;

  const char* Name() const { return "OutIndexTupleExprSignature"; }
};

template <>
struct TypeImpl<index_expr::OpIndexTupleExprSignature> : public std::monostate {
  using value_type = index_expr::OpIndexTupleExprSignature;

  const char* Name() const { return "OpIndexTupleExprSignature"; }
};

}  // namespace ap::axpr
