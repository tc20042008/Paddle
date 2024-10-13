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
#include "ap/kernel_define/func_declare.h"
#include "ap/kernel_define/kernel_arg.h"

namespace ap::kernel_define {

template <typename ValueT>
struct FuncDeclareMethodClass {};

template <typename ValueT>
struct TypeImplFuncDeclareMethodClass {
  using This = TypeImplFuncDeclareMethodClass;
  using Self = axpr::TypeImpl<FuncDeclare>;

  adt::Result<ValueT> Call(const Self&) { return &This::Construct; }

  static adt::Result<ValueT> Construct(const ValueT&,
                                       const std::vector<ValueT>& args) {
    return This{}.Make(args);
  }

  adt::Result<ValueT> Make(const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 2) << adt::errors::TypeError{
        std::string("the constructor of FuncDeclare takes 2 arguments but ") +
        std::to_string(args.size()) + "were given."};
    ADT_LET_CONST_REF(func_id, axpr::TryGetImpl<std::string>(args.at(0)))
        << adt::errors::TypeError{std::string() +
                                  "the argument 1 of constructor of "
                                  "FuncDeclare should be a 'str'"};
    ADT_LET_CONST_REF(kernel_args, GetKernelArgs(args.at(1)));
    return FuncDeclare{func_id, kernel_args};
  }

  Result<adt::List<KernelArg>> GetKernelArgs(const ValueT& val) {
    ADT_LET_CONST_REF(list, axpr::TryGetImpl<adt::List<ValueT>>(val))
        << adt::errors::TypeError{
               std::string() +
               "the argument 2 of constructor of FuncDeclare should be a list "
               "of 'KernelArg's."};
    adt::List<KernelArg> ret;
    ret->reserve(list->size());
    for (const auto& elt : *list) {
      ADT_LET_CONST_REF(kernel_arg, axpr::TryGetImpl<KernelArg>(elt))
          << adt::errors::TypeError{
                 std::string() +
                 "the argument 2 of constructor of FuncDeclare should be a "
                 "list of 'KernelArg's."};
      ret->emplace_back(kernel_arg);
    }
    return ret;
  }
};

}  // namespace ap::kernel_define

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, ap::kernel_define::FuncDeclare>
    : public kernel_define::FuncDeclareMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<ap::kernel_define::FuncDeclare>>
    : public kernel_define::TypeImplFuncDeclareMethodClass<ValueT> {};

}  // namespace ap::axpr
