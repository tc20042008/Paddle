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

#include <functional>
#include <optional>
#include <type_traits>
#include <vector>
#include "paddle/cinn/adt/adt.h"
#include "paddle/common/overloaded.h"

namespace adt = ::cinn::adt;

namespace pexpr {

DEFINE_ADT_TAG(tVar);

struct PrimitiveOp {
  std::string op_name;

  bool operator==(const PrimitiveOp& other) const {
    return this->op_name == other.op_name;
  }
};

template <typename Expr>
struct LambdaImpl {
  std::vector<tVar<std::string>> args;
  Expr body;

  bool operator==(const LambdaImpl& other) const {
    return (this->args == other.args) && (this->body == other.body);
  }
};

template <typename Expr>
DEFINE_ADT_RC(Lambda, const LambdaImpl<Expr>);

// aexpr := Var | CONST | (lambda [VAR] expr)

template <typename Expr>
using AtomicBase = std::variant<tVar<std::string>,
                                bool,
                                int64_t,
                                std::string,
                                PrimitiveOp,
                                Lambda<Expr>>;

template <typename Expr>
struct Atomic : public AtomicBase<Expr> {
  using AtomicBase<Expr>::AtomicBase;
  DEFINE_ADT_VARIANT_METHODS(AtomicBase<Expr>);
};

template <typename Expr>
struct CallImpl {
  Atomic<Expr> func;
  std::vector<Atomic<Expr>> args;

  bool operator==(const CallImpl& other) const {
    return (this->func == other.func) && (this->args == other.args);
  }
};

template <typename Expr>
DEFINE_ADT_RC(Call, const CallImpl<Expr>);

}  // namespace pexpr
