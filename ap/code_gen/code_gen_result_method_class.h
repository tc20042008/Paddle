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
#include "ap/code_gen/code_gen_result.h"

namespace ap::code_gen {

template <typename ValueT>
struct CodeGenResultMethodClass {
  using This = CodeGenResultMethodClass;
  using Self = CodeGenResult<ValueT>;
};

template <typename ValueT>
struct TypeImplCodeGenResultMethodClass {
  using This = TypeImplCodeGenResultMethodClass;
  using Self = axpr::TypeImpl<CodeGenResult<ValueT>>;

  adt::Result<ValueT> Call(const Self&) { return &This::Construct; }

  static adt::Result<ValueT> Construct(const ValueT&,
                                       const std::vector<ValueT>& args) {
    return This{}.Make(args);
  }

  adt::Result<ValueT> Make(const std::vector<ValueT>& packed_args_val) {
    const auto& packed_args = axpr::CastToPackedArgs(packed_args_val);
    const auto& [args, kwargs] = *packed_args;
    ADT_LET_CONST_REF(module_val, kwargs->Get("module"))
        << adt::errors::TypeError{
               std::string() +
               "the constructor of 'CodeGenResult' missing keyword argument "
               "'module' of type 'Module'."};
    ADT_LET_CONST_REF(
        m, axpr::TryGetBuiltinClassInstance<code_module::Module>(module_val))
        << adt::errors::TypeError{
               std::string() +
               "the constructor of 'CodeGenResult' missing keyword argument "
               "'module' of type 'Module'."};
    ADT_LET_CONST_REF(
        kernel_dispatch_func,
        kwargs->template TryGet<axpr::Function<axpr::SerializableValue>>(
            "kernel_dispatch_func"))
        << adt::errors::TypeError{
               std::string() +
               "the constructor of 'CodeGenResult' missing keyword argument "
               "'kernel_dispatch_func' of type 'Function'."};
    std::optional<axpr::AttrMap<axpr::SerializableValue>>
        kernel_dispatch_const_data;
    if (kwargs->Has("kernel_dispatch_const_data")) {
      ADT_LET_CONST_REF(
          data,
          kwargs->template TryGet<axpr::AttrMap<axpr::SerializableValue>>(
              "kernel_dispatch_const_data"))
          << adt::errors::TypeError{
                 std::string() +
                 "the constructor of 'CodeGenResult' needs keyword argument "
                 "'kernel_dispatch_const_data' of type "
                 "'BuiltinSerializableAttrMap'."};
      kernel_dispatch_const_data = data;
    } else {
      kernel_dispatch_const_data = axpr::AttrMap<axpr::SerializableValue>{};
    }
    ADT_CHECK(kernel_dispatch_const_data.has_value());
    return CodeGenResult<ValueT>{
        m, kernel_dispatch_func, kernel_dispatch_const_data.value()};
  }
};

}  // namespace ap::code_gen

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, ap::code_gen::CodeGenResult<ValueT>>
    : public code_gen::CodeGenResultMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<ap::code_gen::CodeGenResult<ValueT>>>
    : public code_gen::TypeImplCodeGenResultMethodClass<ValueT> {};

}  // namespace ap::axpr
