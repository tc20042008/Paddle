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
#include "paddle/pir/include/dialect/pexpr/index_expr.h"
#include "paddle/pir/include/dialect/pexpr/index_tuple_expr.h"
#include "paddle/pir/include/dialect/pexpr/op_signature.h"
#include "paddle/pir/include/dialect/pexpr/type.h"

namespace pexpr {

using InIndexTupleExprSignature = InputSignature<IndexTupleExpr>;
using OutIndexTupleExprSignature = OutputSignature<IndexTupleExpr>;
using OpIndexTupleExprSignature = OpSignature<IndexTupleExpr>;

template <>
struct TypeImpl<InIndexTupleExprSignature> : public std::monostate {
  using value_type = InIndexTupleExprSignature;

  const char* Name() const { return "InIndexTupleExprSignature"; }
};

template <>
struct TypeImpl<OutIndexTupleExprSignature> : public std::monostate {
  using value_type = OutIndexTupleExprSignature;

  const char* Name() const { return "OutIndexTupleExprSignature"; }
};

template <>
struct TypeImpl<OpIndexTupleExprSignature> : public std::monostate {
  using value_type = OpIndexTupleExprSignature;

  const char* Name() const { return "OpIndexTupleExprSignature"; }
};

}  // namespace pexpr
