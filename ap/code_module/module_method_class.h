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
struct ModuleMethodClass {
  using This = ModuleMethodClass;
  using Self = Module;
};

template <typename ValueT>
struct TypeImplModuleMethodClass {
  using This = TypeImplModuleMethodClass;
  using Self = axpr::TypeImpl<Module>;

  adt::Result<ValueT> Call(const Self&) { return &This::Construct; }

  static adt::Result<ValueT> Construct(const ValueT&,
                                       const std::vector<ValueT>& args) {
    return This{}.Make(args);
  }

  adt::Result<ValueT> Make(const std::vector<ValueT>& args) {
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
      ADT_LET_CONST_REF(func_declare, axpr::TryGetImpl<FuncDeclare>(elt))
          << adt::errors::TypeError{
                 std::string() +
                 "the argument 1 of constructor of 'Module' should be a "
                 "'FuncDeclare' object or a list of 'FuncDeclare' object."};
      func_declares->emplace_back(func_declare);
    }
    ADT_LET_CONST_REF(
        source_code_instance,
        args.at(1).template TryGet<axpr::BuiltinClassInstance<ValueT>>())
        << adt::errors::TypeError{
               std::string() +
               "the argument 2 of Module() should be a 'SourceCode' (not " +
               axpr::GetTypeName(args.at(1)) + ") object"};
    ADT_LET_CONST_REF(source_code,
                      source_code_instance->template TryGet<SourceCode>())
        << adt::errors::TypeError{std::string() +
                                  "the argument 1 of constructor of 'Module' "
                                  "should be a 'SourceCode' object."};
    return Module{func_declares, source_code};
  }
};

}  // namespace ap::code_module

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, ap::code_module::Module>
    : public code_module::ModuleMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<ap::code_module::Module>>
    : public code_module::TypeImplModuleMethodClass<ValueT> {};

}  // namespace ap::axpr
