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
#include "ap/axpr/serializable_value_helper.h"

namespace ap::axpr {

template <typename ValueT>
struct BuiltinSerializableAttributeMethodClass {
  using This = BuiltinSerializableAttributeMethodClass;
  using Self = Attribute<SerializableValue>;

  adt::Result<ValueT> Length(const Self& self) {
    return static_cast<int64_t>(self->size());
  }

  adt::Result<ValueT> ToString(const Self& self) {
    ADT_LET_CONST_REF(str, SerializableValueHelper{}.ToString(self));
    return str;
  }

  adt::Result<ValueT> Hash(const Self& self) {
    ADT_LET_CONST_REF(hash_value, SerializableValueHelper{}.Hash(self));
    return hash_value;
  }

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    ADT_LET_CONST_REF(val, self->Get(attr_name)) << adt::errors::AttributeError{
        std::string() + "'BuiltinSerializableAttribute' has no attribute '" +
        attr_name + "'."};
    return val.template CastTo<ValueT>();
  }
};

template <typename ValueT>
struct TypeImplBuiltinSerializableAttributeMethodClass {
  using This = TypeImplBuiltinSerializableAttributeMethodClass;
  using Self = TypeImpl<Attribute<SerializableValue>>;

  adt::Result<ValueT> Call(const Self&) { return &This::StaticConstruct; }

  static adt::Result<ValueT> StaticConstruct(const ValueT&,
                                             const std::vector<ValueT>& args) {
    return This{}.Construct(args);
  }

  adt::Result<ValueT> Construct(const std::vector<ValueT>& args) {
    const auto& packed_args = CastToPackedArgs(args);
    const auto& [pos_args, kwargs] = *packed_args;
    ADT_CHECK(pos_args->empty()) << adt::errors::TypeError{
        std::string() +
        "the construct of BuiltinSerializableAttribute "
        "takes no positional arguments."};
    ADT_LET_CONST_REF(serializable_val,
                      SerializableValueHelper{}.CastObjectFrom(kwargs));
    ADT_LET_CONST_REF(
        serializable_obj,
        serializable_val.template TryGet<Attribute<SerializableValue>>());
    return serializable_obj;
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, Attribute<SerializableValue>>
    : public BuiltinSerializableAttributeMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<Attribute<SerializableValue>>>
    : public TypeImplBuiltinSerializableAttributeMethodClass<ValueT> {};

}  // namespace ap::axpr
