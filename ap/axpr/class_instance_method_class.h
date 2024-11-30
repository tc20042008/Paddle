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

#include "ap/axpr/builtin_high_order_func_type.h"
#include "ap/axpr/class_attrs_helper.h"
#include "ap/axpr/class_instance.h"
#include "ap/axpr/core_expr.h"
#include "ap/axpr/method.h"
#include "ap/axpr/method_class.h"

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, ClassInstance<ValueT>> {
  using Val = ValueT;
  using Self = ClassInstance<ValueT>;
  using This = MethodClassImpl<ValueT, Self>;

  adt::Result<ValueT> Hash(const Self& self) {
    return adt::errors::TypeError{
        "objects are not hashable even __hash__ provided"};
  }

  adt::Result<ValueT> ToString(const Self& self) {
    std::ostringstream ss;
    ADT_LET_CONST_REF(ptr, self->instance_attrs.Get());
    ss << "<" << self->type.class_attrs->class_name << " object at " << ptr
       << ">";
    return ss.str();
  }

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    return GetInstanceOrClassAttr(self, attr_name);
  }

  adt::Result<ValueT> Call(const Self& self) {
    return GetInstanceOrClassAttr(self, "__call__");
  }

  adt::Result<ValueT> GetInstanceOrClassAttr(const Self& self,
                                             const std::string& attr_name) {
    ADT_LET_CONST_REF(instance_attrs, self->instance_attrs.Get());
    if (instance_attrs->Has(attr_name)) {
      return instance_attrs->Get(attr_name);
    }
    const auto& class_attrs = self->type.class_attrs;
    const auto& opt_func = ClassAttrsHelper<ValueT, SerializableValue>{}.OptGet(
        class_attrs, attr_name);
    ADT_CHECK(opt_func.has_value()) << adt::errors::AttributeError{
        std::string() + "type object '" + class_attrs->class_name +
        "' has no attribute '" + attr_name + "'"};
    using RetT = adt::Result<ValueT>;
    return opt_func.value().Match(
        [&](const Function<SerializableValue>& f) -> RetT {
          return Method<ValueT>{self, f};
        },
        [&](const auto&) -> RetT { return opt_func.value(); });
  }

  adt::Result<ValueT> SetAttr(const Self& self, const ValueT& attr_name_val) {
    return Method<ValueT>{self, &This::SetInstanceAttr};
  }

  static adt::Result<ValueT> SetInstanceAttr(const ValueT& self_val,
                                             const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template TryGet<Self>())
        << adt::errors::TypeError{
               std::string() +
               "type(self) is unexpected. given: " + GetTypeName(self_val)};
    ADT_CHECK(args.size() == 2);
    ADT_LET_CONST_REF(attr_name, args.at(0).template TryGet<std::string>());
    ADT_LET_CONST_REF(instance_attrs, self->instance_attrs.Mut());
    instance_attrs->Set(attr_name, args.at(1));
    return adt::Nothing{};
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<ClassInstance<ValueT>>> {
  using Val = ValueT;
  using Self = TypeImpl<ClassInstance<ValueT>>;
  using This = MethodClassImpl<ValueT, Self>;

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    ADT_LET_CONST_REF(attr, self.class_attrs->attrs->Get(attr_name))
        << adt::errors::AttributeError{
               std::string() + "type object '" + self.class_attrs->class_name +
               "' has no attribute '" + attr_name + "'"};
    return attr.template CastTo<ValueT>();
  }

  adt::Result<ValueT> Call(const Self& self) {
    ValueT func{&This::StaticConstruct};
    return Method<ValueT>{self, func};
  }

  adt::Result<ValueT> ToString(const Self& self) {
    return std::string() + "<class '" + self.class_attrs->class_name + "'>";
  }

  adt::Result<ValueT> Hash(const Self& self) {
    return reinterpret_cast<int64_t>(self.class_attrs.shared_ptr().get());
  }

  static adt::Result<ValueT> StaticConstruct(
      axpr::InterpreterBase<ValueT>* interpreter,
      const ValueT& self_val,
      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, axpr::TryGetTypeImpl<Self>(self_val));
    return This{}.Construct(interpreter, self, args);
  }

  adt::Result<ValueT> Construct(axpr::InterpreterBase<ValueT>* interpreter,
                                const Self& self,
                                const std::vector<ValueT>& args) {
    const auto& class_attrs = self.class_attrs;
    const auto& instance = [&] {
      const auto& instance_attrs = InstanceAttrs<ValueT>::Make(
          interpreter->circlable_ref_list(),
          std::make_shared<AttributeImpl<ValueT>>());
      TypeImpl<ClassInstance<ValueT>> type(class_attrs);
      return ClassInstance<ValueT>{type, instance_attrs};
    }();
    const auto& init_func =
        ClassAttrsHelper<ValueT, SerializableValue>{}.OptGet(class_attrs,
                                                             "__init__");
    if (init_func.has_value()) {
      Method<ValueT> f{instance, init_func.value()};
      ADT_RETURN_IF_ERR(interpreter->InterpretCall(f, args));
    } else {
      ADT_CHECK(args.size() == 0) << adt::errors::TypeError{
          std::string() + self.class_attrs->class_name +
          "() takes no arguments"};
    }
    return instance;
  }
};

}  // namespace ap::axpr
