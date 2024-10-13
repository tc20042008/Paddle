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

#include "ap/paddle/pir_node.h"

namespace ap::paddle {

template <typename ValueT>
struct NativeIrValueMethodClass {
  using This = NativeIrValueMethodClass;
  using Self = NativeIrValue;
};

template <typename ValueT>
struct PackedIrValueMethodClass {
  using This = PackedIrValueMethodClass;
  using Self = PackedIrValue;
};

template <typename ValueT>
struct NativeIrOpMethodClass {
  using This = NativeIrOpMethodClass;
  using Self = NativeIrOp;
};

template <typename ValueT>
struct PackedIrOpMethodClass {
  using This = PackedIrOpMethodClass;
  using Self = PackedIrOp;
};

}  // namespace ap::paddle

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, ap::paddle::NativeIrValue>
    : public paddle::NativeIrValueMethodClass<ValueT> {};
template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<ap::paddle::NativeIrValue>> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, ap::paddle::PackedIrValue>
    : public paddle::PackedIrValueMethodClass<ValueT> {};
template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<ap::paddle::PackedIrValue>> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, ap::paddle::NativeIrOp>
    : public paddle::NativeIrOpMethodClass<ValueT> {};
template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<ap::paddle::NativeIrOp>> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, ap::paddle::PackedIrOp>
    : public paddle::PackedIrOpMethodClass<ValueT> {};
template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<ap::paddle::PackedIrOp>> {};

}  // namespace ap::axpr
