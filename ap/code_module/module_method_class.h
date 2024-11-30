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
#include "ap/code_module/func_declare.h"
#include "ap/code_module/module.h"
#include "ap/code_module/source_code.h"

namespace ap::code_module {

template <typename ValueT>
struct TypeImplModuleMethodClass {
  using This = TypeImplModuleMethodClass;
  using Self = axpr::TypeImpl<Module>;

  static adt::Result<Module> Make(const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 2) << adt::errors::TypeError{
        std::string("the constructor of 'Module' takes 2 arguments. but ") +
        std::to_string(args.size()) + "were given."};
    const auto& list = args.at(0).Match(
        [&](const adt::List<ValueT>& l) -> adt::List<ValueT> { return l; },
        [&](const auto& impl) -> adt::List<ValueT> {
          return adt::List<ValueT>{ValueT{impl}};
        });
    adt::List<FuncDeclare> func_declares;
    func_declares->reserve(list->size());
    for (const auto& elt : *list) {
      ADT_LET_CONST_REF(func_declare,
                        axpr::TryGetBuiltinClassInstance<FuncDeclare>(elt))
          << adt::errors::TypeError{
                 std::string() +
                 "the argument 1 of constructor of 'Module' should be a "
                 "'FuncDeclare' object or a list of 'FuncDeclare' object."};
      func_declares->emplace_back(func_declare);
    }
    ADT_LET_CONST_REF(source_code,
                      axpr::TryGetBuiltinClassInstance<SourceCode>(args.at(1)))
        << adt::errors::TypeError{
               std::string() +
               "the argument 2 of Module() should be a 'SourceCode' (not " +
               axpr::GetTypeName(args.at(1)) + ") object"};
    return Module{func_declares, source_code};
  }
};

template <typename ValueT>
adt::Result<ValueT> InitModule(const ValueT& self_val,
                               const std::vector<ValueT>& args) {
  ADT_LET_CONST_REF(
      instance, self_val.template TryGet<axpr::BuiltinClassInstance<ValueT>>());
  ADT_LET_CONST_REF(m, TypeImplModuleMethodClass<ValueT>::Make(args));
  instance.shared_ptr()->instance = m;
  return adt::Nothing{};
}

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> MakeModuleClass() {
  using ClassT = axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>;
  static ClassT cls(axpr::MakeBuiltinClass<ValueT>(
      "Module",
      [&](const auto& DoEach) { DoEach("__init__", &InitModule<ValueT>); }));
  return cls;
}
}  // namespace ap::code_module
