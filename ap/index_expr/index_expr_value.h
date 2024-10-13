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
#include "ap/axpr/builtin_functions.h"
#include "ap/axpr/core_expr.h"
#include "ap/axpr/dim_expr.h"
#include "ap/axpr/value.h"
#include "ap/index_expr/index_expr.h"
#include "ap/index_expr/index_tuple_expr.h"
#include "ap/index_expr/op_index_tuple_expr_signature.h"
#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/dialect/shape/utils/shape_or_data_expr.h"

namespace ap::index_expr {

template <typename ValueT>
using ValueImpl = axpr::ValueBase<ValueT,
                                  symbol::DimExpr,
                                  Slice,
                                  IndexExpr,
                                  IndexTupleExpr,
                                  InIndexTupleExprSignature,
                                  OutIndexTupleExprSignature,
                                  OpIndexTupleExprSignature>;

struct Value : public ValueImpl<Value> {
  using ValueImpl<Value>::ValueImpl;
  DEFINE_ADT_VARIANT_METHODS(ValueImpl<Value>);

  static axpr::Object<Value> GetExportedTypes() {
    return axpr::GetObjectTypeName2Type<Value,
                                        symbol::DimExpr,
                                        Slice,
                                        IndexExpr,
                                        IndexTupleExpr,
                                        InIndexTupleExprSignature,
                                        OutIndexTupleExprSignature,
                                        OpIndexTupleExprSignature>();
  }
};

using Val = Value;

using Env = axpr::Environment<Val>;

using EnvMgr = axpr::EnvironmentManager<Val>;

}  // namespace ap::index_expr
