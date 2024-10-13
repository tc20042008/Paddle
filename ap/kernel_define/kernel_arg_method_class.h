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
#include "ap/kernel_define/kernel_arg.h"

namespace ap::kernel_define {

template <typename ValueT>
struct KernelArgMethodClass {
  using This = KernelArgMethodClass;
  using Self = KernelArg;
};

template <typename ValueT>
struct TypeImplKernelArgMethodClass {
  using This = TypeImplKernelArgMethodClass;
  using Self = axpr::TypeImpl<KernelArg>;

  adt::Result<ValueT> Call(const Self&) { return &This::Construct; }

  static adt::Result<ValueT> Construct(const ValueT&,
                                       const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 2) << adt::errors::TypeError{
        std::string() + "the construct of KernelArg takes 2 arguments. but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(arg_type, CastToArgType(args.at(0)))
        << adt::errors::TypeError{
               std::string() +
               "the argument 1 of construct of KernelArg should be a DataType "
               "object or PointerType object."};
    ADT_LET_CONST_REF(
        lambda, axpr::TryGetImpl<axpr::Lambda<axpr::CoreExpr>>(args.at(1)))
        << adt::errors::TypeError{std::string() +
                                  "the argument 2 of construct of KernelArg "
                                  "should be a function_code object."};
    return KernelArg{arg_type, lambda};
  }
};

}  // namespace ap::kernel_define

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, ap::kernel_define::KernelArg>
    : public kernel_define::KernelArgMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<ap::kernel_define::KernelArg>>
    : public kernel_define::TypeImplKernelArgMethodClass<ValueT> {};

}  // namespace ap::axpr
