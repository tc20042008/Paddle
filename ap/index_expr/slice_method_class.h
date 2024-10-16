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
#include "ap/index_expr/slice.h"

namespace ap::index_expr {

template <typename ValueT>
struct SliceMethodClass {
  using This = SliceMethodClass;
  using Self = Slice;

  adt::Result<ValueT> ToString(const Self& self) { return self->ToString(); }
};

template <typename ValueT>
struct TypeImplSliceMethodClass {
  using This = TypeImplSliceMethodClass;
  using Self = axpr::TypeImpl<Slice>;
};

}  // namespace ap::index_expr

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, index_expr::Slice>
    : public index_expr::SliceMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<index_expr::Slice>>
    : public index_expr::TypeImplSliceMethodClass<ValueT> {};

}  // namespace ap::axpr
