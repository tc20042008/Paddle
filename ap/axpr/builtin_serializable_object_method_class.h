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

#include <sstream>
#include "ap/axpr/builtin_serializable_object.h"
#include "ap/axpr/class_instance.h"
#include "ap/axpr/constants.h"
#include "ap/axpr/method_class.h"
#include "ap/axpr/packed_args.h"

namespace ap::axpr {

template <typename ValueT>
struct BuiltinSerializableObjectMethodClass {
  using This = BuiltinSerializableObjectMethodClass;
  using Self = BuiltinSerializableObject<ValueT>;

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    ADT_LET_CONST_REF(val, self->object->Get(attr_name))
        << adt::errors::AttributeError{
               std::string() +
               "'BuiltinSerializableObject' has no attribute '" + attr_name +
               "'."};
    return val;
  }
};

template <typename ValueT>
struct TypeImplBuiltinSerializableObjectMethodClass {
  using This = TypeImplBuiltinSerializableObjectMethodClass;
  using Self = TypeImpl<BuiltinSerializableObject<ValueT>>;

  adt::Result<ValueT> Call(const Self&) { return &This::StaticConstruct; }

  static adt::Result<ValueT> StaticConstruct(const ValueT&,
                                             const std::vector<ValueT>& args) {
    return This{}.Construct(args);
  }

  adt::Result<ValueT> Construct(const std::vector<ValueT>& args) {
    const auto& packed_args = CastToPackedArgs(args);
    const auto& [pos_args, kwargs] = *packed_args;
    ADT_CHECK(pos_args->empty())
        << adt::errors::TypeError{std::string() +
                                  "the construct of BuiltinSerializableObject "
                                  "takes no positional arguments."};
    ADT_RETURN_IF_ERR(CheckObjectIsBuiltinSerializable(kwargs));
    return BuiltinSerializableObject<ValueT>{kwargs};
  }

  adt::Result<adt::Ok> CheckObjectIsBuiltinSerializable(
      const BuiltinObject<ValueT>& kwargs) {
    for (const auto& [_, val] : kwargs->storage) {
      ADT_RETURN_IF_ERR(CheckValueIsBuiltinSerializable(val));
    }
    return adt::Ok{};
  }

  adt::Result<adt::Ok> CheckValueIsBuiltinSerializable(const ValueT& val) {
    using Ok = adt::Result<adt::Ok>;
    using TypeT = typename TypeTrait<ValueT>::TypeT;
    return val.Match(
        [&](const TypeT& type) -> Ok {
          return type.Match(
              [](const TypeImpl<adt::Nothing>&) -> Ok { return adt::Ok{}; },
              [](const TypeImpl<bool>&) -> Ok { return adt::Ok{}; },
              [](const TypeImpl<int64_t>&) -> Ok { return adt::Ok{}; },
              [](const TypeImpl<double>&) -> Ok { return adt::Ok{}; },
              [](const TypeImpl<std::string>&) -> Ok { return adt::Ok{}; },
              [](const TypeImpl<ClassInstance<ValueT>>&) -> Ok {
                return adt::Ok{};
              },
              [](const TypeImpl<BuiltinSerializableObject<ValueT>>&) -> Ok {
                return adt::Ok{};
              },
              [&](const auto&) -> Ok {
                std::ostringstream ss;
                ss << "Builtin serializable types are: NoneType, bool, int, "
                      "float, "
                      "str, class, BuiltinSerializableObject (not "
                      "include '"
                   << axpr::GetTypeName(val) << "').";
                return adt::errors::ValueError{ss.str()};
              });
        },
        [](const Nothing&) -> Ok { return adt::Ok{}; },
        [](bool) -> Ok { return adt::Ok{}; },
        [](int64_t) -> Ok { return adt::Ok{}; },
        [](double) -> Ok { return adt::Ok{}; },
        [](const std::string&) -> Ok { return adt::Ok{}; },
        [](const Lambda<CoreExpr>&) -> Ok { return adt::Ok{}; },
        [&](const adt::List<ValueT>& list) -> Ok {
          for (const auto& elt : *list) {
            ADT_RETURN_IF_ERR(CheckValueIsBuiltinSerializable(elt));
          }
          return adt::Ok{};
        },
        [&](const BuiltinSerializableObject<ValueT>& object) -> Ok {
          ADT_RETURN_IF_ERR(CheckObjectIsBuiltinSerializable(object->object));
          return adt::Ok{};
        },
        [&](const auto&) -> Ok {
          std::ostringstream ss;
          ss << "Builtin serializable objects are: NoneType, bool, int, float, "
                "str, function_code, list, BuiltinSerializableObject (not "
                "include '"
             << axpr::GetTypeName(val) << "').";
          return adt::errors::ValueError{ss.str()};
        });
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, BuiltinSerializableObject<ValueT>>
    : public BuiltinSerializableObjectMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<BuiltinSerializableObject<ValueT>>>
    : public TypeImplBuiltinSerializableObjectMethodClass<ValueT> {};

}  // namespace ap::axpr
