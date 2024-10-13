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
#include "ap/kernel_define/source_code.h"

namespace ap::kernel_define {

template <typename ValueT>
struct SourceCodeMethodClass {
  using This = SourceCodeMethodClass;
  using Self = SourceCode;
};

template <typename ValueT>
struct TypeImplSourceCodeMethodClass {
  using This = TypeImplSourceCodeMethodClass;
  using Self = axpr::TypeImpl<SourceCode>;

  adt::Result<ValueT> Call(const Self&) { return &This::Construct; }

  static adt::Result<ValueT> Construct(const ValueT&,
                                       const std::vector<ValueT>& args) {
    return This{}.Make(args);
  }

  adt::Result<ValueT> Make(const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string("the constructor of 'SourceCode' takes 1 arguments. but ") +
        std::to_string(args.size()) + "were given."};
    ADT_LET_CONST_REF(str, axpr::TryGetImpl<std::string>(args.at(0)))
        << adt::errors::TypeError{
               std::string("the argument 1 of constructor of 'SourceCode' must "
                           "be a 'str'.")};
    return SourceCode{str};
  }
};

}  // namespace ap::kernel_define

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, ap::kernel_define::SourceCode>
    : public kernel_define::SourceCodeMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<ap::kernel_define::SourceCode>>
    : public kernel_define::TypeImplSourceCodeMethodClass<ValueT> {};

}  // namespace ap::axpr
