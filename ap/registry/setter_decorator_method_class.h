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
#include "ap/axpr/type.h"
#include "ap/registry/setter_decorator.h"

namespace ap::registry {

template <typename ValueT>
struct SetterDecoratorMethodClass {
  using This = SetterDecoratorMethodClass;
  using Self = SetterDecorator;

  adt::Result<ValueT> Call(const Self& self) {
    return axpr::Method<ValueT>{self, &This::StaticCall};
  }

  using Function = axpr::Function<axpr::SerializableValue>;

  static adt::Result<ValueT> StaticCall(const ValueT& self_val,
                                        const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, axpr::TryGetImpl<Self>(self_val));
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(function, This{}.CastToFunction(args.at(0)));
    self->lambda.shared_ptr()->data = function;
    return adt::Nothing{};
  }

  adt::Result<Function> CastToFunction(const ValueT& val) {
    using RetT = adt::Result<Function>;
    return val.Match(
        [&](const Function& function) -> RetT { return function; },
        [&](const axpr::Closure<ValueT>& closure) -> RetT {
          return Function{closure->lambda, std::nullopt};
        },
        [&](const auto&) -> RetT {
          return adt::errors::TypeError{
              std::string() +
              "decorator must be applied to a function or closure (not " +
              axpr::GetTypeName(val) + ")"};
        });
  }
};

}  // namespace ap::registry

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, registry::SetterDecorator>
    : public registry::SetterDecoratorMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<registry::SetterDecorator>> {};

}  // namespace ap::axpr
