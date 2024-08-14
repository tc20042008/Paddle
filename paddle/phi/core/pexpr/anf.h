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
struct If {
  Atomic<Expr> cond;
  std::shared_ptr<Expr> true_expr;
  std::shared_ptr<Expr> false_expr;
};

template <typename Expr>
using CombinedBase = std::variant<Call<Expr>, If<Expr>>;

template <typename Expr>
struct Combined : public CombinedBase<Expr> {
  using CombinedBase<Expr>::CombinedBase;

  DEFINE_MATCH_METHOD();

  const CombinedBase<Expr>& variant() const {
    return reinterpret_cast<const CombinedBase<Expr>&>(*this);
  }

  template <typename T>
  bool Has() const {
    return std::holds_alternative<T>(variant());
  }

  template <typename T>
  const T& Get() const {
    return std::get<T>(variant());
  }
};

template <typename Expr>
struct Bind {
  tVar<std::string> var;
  Combined<Expr> val;
};

template <typename Expr>
struct Let {
  std::vector<Bind<Expr>> bindings;
  std::shared_ptr<Expr> body;
};

struct AnfExpr;

// expr := aexpr | cexpr | let [VAR cexpr] expr
// cexpr := (aexpr aexpr ...) | (If aexpr expr expr)
using AnfExprBase =
    std::variant<Atomic<AnfExpr>, Combined<AnfExpr>, Let<AnfExpr>>;

// A-norm form
struct AnfExpr : public AnfExprBase {
  using AnfExprBase::AnfExprBase;

  DEFINE_MATCH_METHOD();

  const AnfExprBase& variant() const {
    return reinterpret_cast<const AnfExprBase&>(*this);
  }

  template <typename T>
  bool Has() const {
    return std::holds_alternative<T>(variant());
  }

  template <typename T>
  const T& Get() const {
    return std::get<T>(variant());
  }

  std::string DumpToJsonString();
  static std::optional<AnfExpr> ParseFromJsonString(
      const std::string& json_str);
};

}  // namespace pexpr
