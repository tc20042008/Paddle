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

#include "ap/axpr/method_class.h"
#include "ap/index_expr/op_index_tuple_expr_signature.h"

namespace ap::index_expr {

template <typename ValueT>
struct InIndexTupleExprSignatureMethodClass {};

template <typename ValueT>
struct OutIndexTupleExprSignatureMethodClass {};

template <typename ValueT>
struct OpIndexTupleExprSignatureMethodClass {};

}  // namespace ap::index_expr

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, index_expr::InIndexTupleExprSignature>
    : public index_expr::InIndexTupleExprSignatureMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<index_expr::InIndexTupleExprSignature>>
    : public EmptyMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, index_expr::OutIndexTupleExprSignature>
    : public index_expr::OutIndexTupleExprSignatureMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<index_expr::OutIndexTupleExprSignature>>
    : public EmptyMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, index_expr::OpIndexTupleExprSignature>
    : public index_expr::OpIndexTupleExprSignatureMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<index_expr::OpIndexTupleExprSignature>>
    : public EmptyMethodClass<ValueT> {};

}  // namespace ap::axpr
