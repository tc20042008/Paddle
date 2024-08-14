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
#include <ostream>
#include <vector>
#include "paddle/phi/core/pexpr/atomic.h"

namespace pexpr {

struct CoreExpr;

// (outter_func (inner_func [args]))
template <typename Expr>
struct ComposedCall {
  Atomic<Expr> outter_func;
  Atomic<Expr> inner_func;
  std::vector<Atomic<Expr>> args;

  bool operator==(const ComposedCall& other) const {
    return (this->outter_func == other.outter_func) &&
           (this->inner_func == other.inner_func) && (this->args == other.args);
  }
};

// core expr
// expr := aexpr | (aexpr (aexpr [aexpr]))
using CoreExprBase = std::variant<Atomic<CoreExpr>, ComposedCall<CoreExpr>>;

struct CoreExpr : public CoreExprBase {
  using CoreExprBase::CoreExprBase;

  DEFINE_MATCH_METHOD();

  const CoreExprBase& variant() const {
    return reinterpret_cast<const CoreExprBase&>(*this);
  }

  template <typename T>
  bool Has() const {
    return std::holds_alternative<T>(variant());
  }

  template <typename T>
  const T& Get() const {
    return std::get<T>(variant());
  }

  bool operator==(const CoreExpr& other) const {
    return std::visit(CompareFunctor{}, this->variant(), other.variant());
  }

  std::string ToSExpression() const;
  std::string DumpToJsonString();
  static std::optional<CoreExpr> ParseFromJsonString(
      const std::string& json_str);

 private:
  struct CompareFunctor {
    bool operator()(const Atomic<CoreExpr>& lhs,
                    const Atomic<CoreExpr>& rhs) const {
      return lhs == rhs;
    }
    bool operator()(const ComposedCall<CoreExpr>& lhs,
                    const ComposedCall<CoreExpr>& rhs) const {
      return lhs == rhs;
    }
    bool operator()(const auto& lhs, const auto& rhs) const { return false; }
  };
};

extern const char kBuiltinId[];

}  // namespace pexpr

namespace std {

inline std::ostream& operator<<(std::ostream& os,
                                const pexpr::CoreExpr& core_expr) {
  return os << core_expr.ToSExpression();
}

}  // namespace std
