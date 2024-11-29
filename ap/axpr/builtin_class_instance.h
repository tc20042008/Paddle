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

#include <any>
#include <unordered_map>
#include "ap/axpr/adt.h"
#include "ap/axpr/class_attrs.h"
#include "ap/axpr/error.h"
#include "ap/axpr/serializable_value.h"
#include "ap/axpr/type.h"

namespace ap::axpr {

template <typename ValueT>
struct BuiltinClassInstance;

template <typename ValueT>
struct TypeImpl<BuiltinClassInstance<ValueT>> {
  explicit TypeImpl<BuiltinClassInstance<ValueT>>(
      const ClassAttrs<ValueT>& class_attr_val)
      : class_attrs(class_attr_val) {}

  ClassAttrs<ValueT> class_attrs;

  const std::string& Name() const { return class_attrs->Name(); }

  bool operator==(const TypeImpl<BuiltinClassInstance<ValueT>>& other) const {
    return this->class_attrs == other.class_attrs;
  }
};

template <typename ValueT>
struct BuiltinClassInstanceImpl {
  TypeImpl<BuiltinClassInstance<ValueT>> type;
  std::any instance;

  template <typename T>
  adt::Result<T> TryGet() const {
    try {
      return std::any_cast<T>(this->instance);
    } catch (const std::bad_any_cast& e) {
      return adt::errors::TypeError{std::string() + type.Name() +
                                    " class cast to " + typeid(T).name() +
                                    "failed."};
    }
  }

  bool operator==(const BuiltinClassInstanceImpl& other) const {
    return this == &other;
  }
};

template <typename ValueT>
DEFINE_ADT_RC(BuiltinClassInstance, BuiltinClassInstanceImpl<ValueT>);

template <typename ValueT, typename VisitorT>
TypeImpl<BuiltinClassInstance<ValueT>> MakeBuiltinClass(
    const std::string& class_name, const VisitorT& Visitor) {
  using TypeImplT = TypeImpl<BuiltinClassInstance<ValueT>>;
  AttrMap<ValueT> attr_map;
  Visitor(
      [&](const auto& name, const auto& func) { attr_map->Set(name, func); });
  adt::List<std::shared_ptr<ClassAttrsImpl<ValueT>>> empty_superclasses{};
  ClassAttrs<ValueT> class_attrs{class_name, empty_superclasses, attr_map};
  return TypeImplT(class_attrs);
}

}  // namespace ap::axpr
