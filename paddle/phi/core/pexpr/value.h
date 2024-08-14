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

#include <list>
#include "paddle/phi/core/pexpr/atomic.h"

namespace pexpr {

template <typename Expr, typename CustomT>
class Value;

template <typename Expr, typename CustomT>
struct Frame {
  std::optional<Value<Expr, CustomT>> Get(const std::string& var) const {
    const auto& iter = var2val_.find(var);
    if (iter == var2val_.end()) {
      return std::nullopt;
    }
    return iter->second;
  }

  bool Set(const std::string& var, const Value<Expr, CustomT>& val) {
    return var2val_.insert({var, val}).second;
  }

  void Destory() { var2val_.clear(); }

  std::unordered_map<std::string, Value<Expr, CustomT>> var2val_;
};

template <typename Expr, typename CustomT>
struct EnvironmentManager;

template <typename Expr, typename CustomT>
class Environment {
 public:
  std::optional<Value<Expr, CustomT>> Get(const std::string& var) const {
    if (const auto& value = frame_.Get(var)) {
      return value;
    }
    if (parent_ == nullptr) {
      return std::nullopt;
    }
    return parent_->Get(var);
  }

  bool Set(const std::string& var, const Value<Expr, CustomT>& val) {
    return frame_.Set(var, val);
  }

  void Destory() {
    parent_ = nullptr;
    frame_.Destory();
  }

 private:
  static std::shared_ptr<Environment> New(
      const std::shared_ptr<Environment>& parent) {
    return std::shared_ptr<Environment>(new Environment(parent));
  }

  explicit Environment(const std::shared_ptr<Environment>& parent)
      : parent_(parent), frame_() {}
  Environment(const Environment&) = delete;
  Environment(Environment&&) = delete;

  friend class EnvironmentManager<Expr, CustomT>;

  std::shared_ptr<Environment> parent_;
  Frame<Expr, CustomT> frame_;
};

template <typename Expr, typename CustomT>
struct Closure {
  Lambda<Expr> lambda;
  std::shared_ptr<Environment<Expr, CustomT>> environment;
};

template <typename Expr, typename CustomT>
using ValueBase = std::variant<CustomT,
                               int64_t,
                               std::string,
                               bool,
                               PrimitiveOp,
                               Closure<Expr, CustomT>,
                               adt::List<Value<Expr, CustomT>>>;

template <typename Expr, typename CustomT>
struct Value : public ValueBase<Expr, CustomT> {
  using ValueBase<Expr, CustomT>::ValueBase;

  DEFINE_MATCH_METHOD();

  const ValueBase<Expr, CustomT>& variant() const {
    return reinterpret_cast<const ValueBase<Expr, CustomT>&>(*this);
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

// Watch out circle references:
// Value   ->   Closure
//   /\              |
//   |               V
//  Frame  <-  Environment
// To avoid memory leak, make sure every Environment instances are created by
// EnvironmentManager;
template <typename Expr, typename CustomT>
class EnvironmentManager {
 public:
  EnvironmentManager() {}
  ~EnvironmentManager() {
    for (const auto& weak_env : weak_envs_) {
      if (auto env = weak_env.lock()) {
        env->Destory();
      }
    }
  }

  std::shared_ptr<Environment<Expr, CustomT>> New(
      const std::shared_ptr<Environment<Expr, CustomT>>& parent) {
    auto ptr = Environment<Expr, CustomT>::New(parent);
    weak_envs_.push_back(ptr);
    return ptr;
  }

 private:
  std::list<std::weak_ptr<Environment<Expr, CustomT>>> weak_envs_;
};

}  // namespace pexpr
