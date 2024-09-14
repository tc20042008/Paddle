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

#include <unordered_map>
#include "ap/axpr/adt.h"
#include "ap/axpr/error.h"
#include "ap/axpr/type.h"

namespace ap::axpr {

template <typename ValueT>
struct ObjectImpl {
  std::unordered_map<std::string, ValueT> storage;

  size_t size() const { return storage.size(); }

  void clear() { storage.clear(); }

  Result<ValueT> Get(const std::string& var) const {
    const auto& iter = storage.find(var);
    if (iter == storage.end()) {
      return AttributeError{"object has no attribute '" + var + "'"};
    }
    return iter->second;
  }

  bool Set(const std::string& var, const ValueT& val) {
    return storage.insert({var, val}).second;
  }

  bool operator==(const ObjectImpl& other) const { return &other == this; }
};

template <typename ValueT>
DEFINE_ADT_RC(Object, ObjectImpl<ValueT>);

template <typename ValueT>
struct TypeImpl<Object<ValueT>> : public std::monostate {
  using value_type = Object<ValueT>;

  const char* Name() const { return "object"; }
};

}  // namespace ap::axpr
