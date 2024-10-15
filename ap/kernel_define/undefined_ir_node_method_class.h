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
#include "ap/kernel_define/undefined_ir_node.h"

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, kernel_define::UndefinedDimExpr> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<kernel_define::UndefinedDimExpr>> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, kernel_define::UndefinedNativeIrValue> {};

template <typename ValueT>
struct MethodClassImpl<ValueT,
                       TypeImpl<kernel_define::UndefinedNativeIrValue>> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, kernel_define::UndefinedPackedIrValue> {};

template <typename ValueT>
struct MethodClassImpl<ValueT,
                       TypeImpl<kernel_define::UndefinedPackedIrValue>> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, kernel_define::UndefinedNativeIrOp> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<kernel_define::UndefinedNativeIrOp>> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, kernel_define::UndefinedPackedIrOp> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<kernel_define::UndefinedPackedIrOp>> {};

}  // namespace ap::axpr
