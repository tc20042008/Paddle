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
struct Lambda {
  std::vector<tVar<std::string>> args;
  std::shared_ptr<Expr> body;

  bool operator==(const Lambda& other) const {
    return (this->args == other.args) && (*this->body == *body);
  }
};

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

  DEFINE_MATCH_METHOD();

  const AtomicBase<Expr>& variant() const {
    return reinterpret_cast<const AtomicBase<Expr>&>(*this);
  }

  template <typename T>
  bool Has() const {
    return std::holds_alternative<T>(variant());
  }

  template <typename T>
  const T& Get() const {
    return std::get<T>(variant());
  }

  bool operator==(const Atomic& other) const {
    return std::visit(CompareFunctor{}, this->variant(), other.variant());
  }

 private:
  struct CompareFunctor {
    bool operator()(const tVar<std::string>& lhs,
                    const tVar<std::string>& rhs) const {
      return lhs == rhs;
    }
    bool operator()(const bool lhs, const bool rhs) const { return lhs == rhs; }
    bool operator()(const int64_t lhs, const int64_t rhs) const {
      return lhs == rhs;
    }
    bool operator()(const std::string& lhs, const std::string& rhs) const {
      return lhs == rhs;
    }
    bool operator()(const PrimitiveOp& lhs, const PrimitiveOp& rhs) const {
      return lhs == rhs;
    }
    bool operator()(const Lambda<Expr>& lhs, const Lambda<Expr>& rhs) const {
      return lhs == rhs;
    }
    bool operator()(const auto& lhs, const auto& rhs) const { return false; }
  };
};

template <typename Expr>
struct Call {
  Atomic<Expr> func;
  std::vector<Atomic<Expr>> args;

  bool operator==(const Call& other) const {
    return (this->func == other.func) && (this->args == other.args);
  }
};

}  // namespace pexpr
