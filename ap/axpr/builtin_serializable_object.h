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

#include "ap/adt/adt.h"
#include "ap/axpr/object.h"
#include "ap/axpr/type.h"

namespace ap::axpr {

template <typename ValueT>
struct BuiltinSerializableObjectImpl {
  Object<ValueT> object;

  bool operator==(const BuiltinSerializableObjectImpl& other) const {
    return this->object == other.object;
  }
};

template <typename ValueT>
DEFINE_ADT_RC(BuiltinSerializableObject, BuiltinSerializableObjectImpl<ValueT>);

template <typename ValueT>
struct TypeImpl<BuiltinSerializableObject<ValueT>> : public std::monostate {
  using value_type = BuiltinSerializableObject<ValueT>;

  const char* Name() const { return "BuiltinSerializableObject"; }
};

}  // namespace ap::axpr
