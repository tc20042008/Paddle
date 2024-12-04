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
#include "ap/include/axpr/adt.h"
#include "ap/include/axpr/builtin_func_type.h"
#include "ap/include/axpr/class_attrs.h"
#include "ap/include/axpr/error.h"
#include "ap/include/axpr/serializable_value.h"
#include "ap/include/axpr/type.h"

namespace ap::axpr {

template <typename ValueT>
struct BuiltinClassInstance;

template <typename ValueT>
struct TypeImpl<BuiltinClassInstance<ValueT>> {
  explicit TypeImpl<BuiltinClassInstance<ValueT>>(
      const ClassAttrs<ValueT>& class_attr_val)
      : class_attrs(class_attr_val) {}

  ClassAttrs<ValueT> class_attrs;

  ValueT New(const std::any& any) const;

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
  bool Has() const {
    return this->instance.type() == typeid(T);
  }

  template <typename T>
  adt::Result<T> TryGet() const {
    if (this->template Has<T>()) {
      return std::any_cast<T>(this->instance);
    } else {
      return adt::errors::TypeError{
          std::string() + "casting from " + type.Name() +
          " class (cpp class name: " + instance.type().name() + ") to " +
          typeid(T).name() + " failed."};
    }
  }

  bool operator==(const BuiltinClassInstanceImpl& other) const {
    return this == &other;
  }
};

template <typename ValueT>
DEFINE_ADT_RC(BuiltinClassInstance, BuiltinClassInstanceImpl<ValueT>);

template <typename ValueT>
ValueT TypeImpl<BuiltinClassInstance<ValueT>>::New(const std::any& any) const {
  return BuiltinClassInstance<ValueT>{*this, any};
}

template <typename ValueT, typename VisitorT>
TypeImpl<BuiltinClassInstance<ValueT>> MakeBuiltinClass(
    const std::string& class_name, const VisitorT& Visitor) {
  using TypeImplT = TypeImpl<BuiltinClassInstance<ValueT>>;
  AttrMap<ValueT> attr_map;
  Visitor([&](const auto& name, const axpr::BuiltinFunction<ValueT>& func) {
    attr_map->Set(name, func.template CastTo<ValueT>());
  });
  adt::List<std::shared_ptr<ClassAttrsImpl<ValueT>>> empty_superclasses{};
  ClassAttrs<ValueT> class_attrs{class_name, empty_superclasses, attr_map};
  return TypeImplT(class_attrs);
}

}  // namespace ap::axpr
