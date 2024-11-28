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
#include "ap/axpr/attribute.h"
#include "ap/axpr/bool.h"
#include "ap/axpr/class_attrs.h"
#include "ap/axpr/float.h"
#include "ap/axpr/function.h"
#include "ap/axpr/int.h"
#include "ap/axpr/nothing.h"
#include "ap/axpr/string.h"
#include "ap/axpr/type.h"

namespace ap::axpr {

template <typename SerializableValueT>
using SerializableValueImpl = std::variant<TypeImpl<adt::Nothing>,
                                           TypeImpl<bool>,
                                           TypeImpl<int64_t>,
                                           TypeImpl<double>,
                                           TypeImpl<std::string>,
                                           ClassAttrs<SerializableValueT>,
                                           adt::Nothing,
                                           bool,
                                           int64_t,
                                           double,
                                           std::string,
                                           Function<SerializableValueT>,
                                           adt::List<SerializableValueT>,
                                           Attribute<SerializableValueT>>;

template <typename ValueT>
struct ClassInstance;

struct SerializableValue : public SerializableValueImpl<SerializableValue> {
  using SerializableValueImpl<SerializableValue>::SerializableValueImpl;

  DEFINE_ADT_VARIANT_METHODS(SerializableValueImpl<SerializableValue>);

  template <typename ValueT>
  ValueT CastTo() const {
    return Match(
        [&](const ClassAttrs<SerializableValue>& class_attrs) -> ValueT {
          return TypeImpl<ClassInstance<ValueT>>(class_attrs);
        },
        [&](const auto& impl) -> ValueT { return impl; });
  }

  template <typename ValueT>
  static bool IsSerializable(const ValueT& val) {
    using TypeT = typename TypeTrait<ValueT>::TypeT;
    return val.Match(
        [&](const TypeT& type) -> bool {
          return type.Match(
              [](const TypeImpl<adt::Nothing>&) -> bool { return true; },
              [](const TypeImpl<bool>&) -> bool { return true; },
              [](const TypeImpl<int64_t>&) -> bool { return true; },
              [](const TypeImpl<double>&) -> bool { return true; },
              [](const TypeImpl<std::string>&) -> bool { return true; },
              [](const TypeImpl<ClassInstance<ValueT>>&) -> bool {
                return true;
              },
              [&](const auto&) -> bool { return false; });
        },
        [](const Nothing&) -> bool { return true; },
        [](bool) -> bool { return true; },
        [](int64_t) -> bool { return true; },
        [](double) -> bool { return true; },
        [](const std::string&) -> bool { return true; },
        [](const Function<SerializableValue>&) -> bool { return true; },
        [](const adt::List<SerializableValue>&) -> bool { return true; },
        [](const Attribute<SerializableValue>&) -> bool { return true; },
        [&](const adt::List<ValueT>& list) -> bool {
          for (const auto& elt : *list) {
            if (!IsSerializable(elt)) {
              return false;
            }
          }
          return true;
        },
        [&](const Attribute<ValueT>& object) -> bool {
          for (const auto& [k, v] : object->object->storage) {
            if (!IsSerializable(v)) {
              return false;
            }
          }
          return true;
        },
        [&](const auto&) -> bool { return false; });
  }

  static std::string SerializableTypeNames() {
    return "NoneType, bool, int, float, str, class, function, "
           "BuiltinSerializableList, BuiltinSerializableAttribute";
  }
};

}  // namespace ap::axpr
