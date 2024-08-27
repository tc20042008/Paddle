// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/adt/adt.h"

namespace cinn::adt {

#define DEFINE_ADT_ARITHMETIC(op) \
  template <typename T>           \
  struct op##Impl {               \
    T lhs;                        \
    T rhs;                        \
  };                              \
  template <typename T>           \
  DEFINE_ADT_RC(op, op##Impl<T>);

DEFINE_ADT_ARITHMETIC(Add);
DEFINE_ADT_ARITHMETIC(Sub);
DEFINE_ADT_ARITHMETIC(Mul);
DEFINE_ADT_ARITHMETIC(Div);
DEFINE_ADT_ARITHMETIC(Mod);

template <typename T>
using ArithmeticImpl = std::variant<Add<T>, Sub<T>, Mul<T>, Div<T>, Mod<T>>;

template <typename T>
struct Arithmetic : public ArithmeticImpl<T> {
  using ArithmeticImpl<T>::ArithmeticImpl;
  DEFINE_ADT_VARIANT_METHODS(ArithmeticImpl<T>);
};

}  // namespace cinn::adt
