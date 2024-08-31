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
#include "paddle/cinn/adt/tree.h"

namespace cinn::adt {

#define DEFINE_ADT_COMPARE(op) \
  template <typename T>        \
  struct op##Impl {            \
    T lhs;                     \
    T rhs;                     \
  };                           \
  template <typename T>        \
  DEFINE_ADT_RC(op, op##Impl<T>);

DEFINE_ADT_COMPARE(EQ);
DEFINE_ADT_COMPARE(LT);
DEFINE_ADT_COMPARE(GT);
DEFINE_ADT_COMPARE(NE);
DEFINE_ADT_COMPARE(GE);
DEFINE_ADT_COMPARE(LE);

DEFINE_ADT_COMPARE(And);
DEFINE_ADT_COMPARE(Or);

template <typename T>
struct NotImpl {
  T operand;
};
template <typename T>
DEFINE_ADT_RC(Not, NotImpl<T>);

template <typename T>
using CompareOpImpl = std::variant<EQ<T>, LT<T>, GT<T>, NE<T>, GE<T>, LE<T>>;
template <typename T>
struct CompareOp : public CompareOpImpl<T> {
  using CompareOpImpl<T>::CompareOpImpl;
  DEFINE_ADT_VARIANT_METHODS(CompareOpImpl<T>);
};

template <typename T>
using LogicalOpImpl = std::variant<And<T>, Or<T>, Not<T>>;
template <typename T>
struct LogicalOp : public LogicalOpImpl<T> {
  using LogicalOpImpl<T>::LogicalOpImpl;
  DEFINE_ADT_VARIANT_METHODS(LogicalOpImpl<T>);
};

template <typename T>
struct Logical : public Tree<LogicalOp, CompareOp<T>> {
  using Tree<LogicalOp, CompareOp<T>>::Tree;
};

}  // namespace cinn::adt
