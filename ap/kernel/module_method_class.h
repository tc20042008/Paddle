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
#include "ap/kernel/module.h"

namespace ap::kernel_define {

template <typename ValueT>
struct ModuleMethodClass {
  using Self = ModuleMethodClass;

  template <typename BuiltinUnarySymbol>
  static std::optional<BuiltinUnaryFuncT<ValueT>> GetBuiltinUnaryFunc() {
    return std::nullopt;
  }

  template <typename BultinBinarySymbol>
  static std::optional<BuiltinBinaryFuncT<ValueT>> GetBuiltinBinaryFunc() {
    return std::nullopt;
  }
};

}  // namespace ap::kernel_define

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, ap::kernel_define::Module> {
  using method_class = ap::kernel_define::ModuleMethodClass<ValueT>;

  template <typename BuiltinUnarySymbol>
  static std::optional<BuiltinUnaryFuncT<ValueT>> GetBuiltinUnaryFunc() {
    return method_class::template GetBuiltinUnaryFunc<BuiltinUnarySymbol>();
  }

  template <typename BultinBinarySymbol>
  static std::optional<BuiltinBinaryFuncT<ValueT>> GetBuiltinBinaryFunc() {
    return method_class::template GetBuiltinBinaryFunc<BultinBinarySymbol>();
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<ap::kernel_define::Module>>
    : public EmptyMethodClass<ValueT> {};

}  // namespace ap::axpr
