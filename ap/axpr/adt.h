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
#include "paddle/cinn/adt/adt.h"

namespace ap::axpr {

namespace adt = ::cinn::adt;

DEFINE_ADT_TAG(tVar);

using adt::Nothing;
using adt::Result;

template <typename T>
using Maybe = adt::Maybe<T>;

template <typename T>
struct DisjointUnionImpl {
  T lhs;
  T rhs;

  bool operator==(const DisjointUnionImpl& other) const {
    return (other.lhs == this->lhs) && (other.rhs == this->rhs);
  }
};

template <typename T>
DEFINE_ADT_RC(DisjointUnion, const DisjointUnionImpl<T>);

}  // namespace ap::axpr
