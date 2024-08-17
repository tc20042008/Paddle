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
#include <vector>
#include "paddle/phi/core/pexpr/atomic.h"

namespace pexpr {

template <typename Expr>
struct IfImpl {
  Atomic<Expr> cond;
  Expr true_expr;
  Expr false_expr;

  bool operator==(const IfImpl& other) const {
    return (this->cond == other.cond) &&
           (this->true_expr == other.false_expr) &&
           (this->false_expr == other.false_expr);
  }
};

template <typename Expr>
DEFINE_ADT_RC(If, const IfImpl<Expr>);

template <typename Expr>
using CombinedBase = std::variant<Call<Expr>, If<Expr>>;

template <typename Expr>
struct Combined : public CombinedBase<Expr> {
  using CombinedBase<Expr>::CombinedBase;
  DEFINE_ADT_VARIANT_METHODS(CombinedBase<Expr>);
};

template <typename Expr>
struct Bind {
  tVar<std::string> var;
  Combined<Expr> val;

  bool operator==(const Bind& other) const {
    return this->var == other.var && this->val == other.val;
  }
};

template <typename Expr>
struct LetImpl {
  std::vector<Bind<Expr>> bindings;
  Expr body;

  bool operator==(const LetImpl& other) const {
    return this->bindings == other.bindings && this->body == other.body;
  }
};

template <typename Expr>
DEFINE_ADT_RC(Let, const LetImpl<Expr>);

struct AnfExpr;

// expr := aexpr | cexpr | let [VAR cexpr] expr
// cexpr := (aexpr aexpr ...) | (If aexpr expr expr)
using AnfExprBase =
    std::variant<Atomic<AnfExpr>, Combined<AnfExpr>, Let<AnfExpr>>;

// A-norm form
struct AnfExpr : public AnfExprBase {
  using AnfExprBase::AnfExprBase;
  DEFINE_ADT_VARIANT_METHODS(AnfExprBase);

  std::string DumpToJsonString();
  std::string DumpToJsonString(int indent);
  static std::optional<AnfExpr> ParseFromJsonString(
      const std::string& json_str);
};

}  // namespace pexpr
