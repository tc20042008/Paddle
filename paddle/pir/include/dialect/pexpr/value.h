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
#include "paddle/pir/include/dialect/pexpr/adt.h"
#include "paddle/pir/include/dialect/pexpr/builtin_func_type.h"
#include "paddle/pir/include/dialect/pexpr/builtin_symbol.h"
#include "paddle/pir/include/dialect/pexpr/closure.h"
#include "paddle/pir/include/dialect/pexpr/cps_builtin_high_order_func_type.h"
#include "paddle/pir/include/dialect/pexpr/data_type.h"
#include "paddle/pir/include/dialect/pexpr/data_value.h"
#include "paddle/pir/include/dialect/pexpr/error.h"
#include "paddle/pir/include/dialect/pexpr/method.h"
#include "paddle/pir/include/dialect/pexpr/object.h"
#include "paddle/pir/include/dialect/pexpr/pointer_type.h"
#include "paddle/pir/include/dialect/pexpr/pointer_value.h"

namespace pexpr {

using adt::Nothing;

template <typename CustomT>
class Value;

template <typename ValueT>
struct Frame {
  Result<ValueT> Get(const std::string& var) const {
    return frame_obj->Get(var);
  }

  bool Set(const std::string& var, const ValueT& val) {
    return frame_obj->Set(var, val);
  }

  bool HasVar(const std::string& var) const {
    return frame_obj->find(var) != frame_obj->end();
  }

  void ClearFrame() { frame_obj->clear(); }

  Object<ValueT> frame_obj;
};

template <typename ValueT>
struct EnvironmentManager;

template <typename ValueT>
class Environment {
 public:
  Result<ValueT> Get(const std::string& var) const {
    const Result<ValueT>& res = frame_.Get(var);
    if (res.template Has<ValueT>()) {
      return res;
    }
    if (parent_ == nullptr) {
      return NameError{std::string("name '") + var + "' is not defined."};
    }
    return parent_->Get(var);
  }

  bool Set(const std::string& var, const ValueT& val) {
    return frame_.Set(var, val);
  }

  void ClearFrame() {
    parent_ = nullptr;
    frame_.ClearFrame();
  }

 private:
  static std::shared_ptr<Environment> New(
      const std::shared_ptr<Environment>& parent) {
    return std::shared_ptr<Environment>(
        new Environment(parent, Frame<ValueT>{}));
  }

  static std::shared_ptr<Environment> NewInitEnv(const Frame<ValueT>& frame) {
    return std::shared_ptr<Environment>(new Environment(nullptr, frame));
  }

  explicit Environment(const std::shared_ptr<Environment>& parent,
                       const Frame<ValueT>& frame)
      : parent_(parent), frame_(frame) {}

  Environment(const Environment&) = delete;
  Environment(Environment&&) = delete;

  friend class EnvironmentManager<ValueT>;

  std::shared_ptr<Environment> parent_;
  Frame<ValueT> frame_;
};

template <typename ValueT, typename... CustomTs>
using ValueBase = std::variant<Nothing,
                               DataType,
                               PointerType,
                               DataValue,
                               PointerValue,
                               std::string,
                               Closure<ValueT>,
                               Method<ValueT>,
                               adt::List<ValueT>,
                               Object<ValueT>,
                               builtin_symbol::Symbol,
                               BuiltinFuncType<ValueT>,
                               CpsBuiltinHighOrderFuncType<ValueT>,
                               CustomTs...>;

template <typename ValueT>
using Builtin = ValueBase<ValueT>;

// Watch out circle references:
// Value   ->   Closure
//   /\              |
//   |               V
//  Frame  <-  Environment
// To avoid memory leak, make sure every Environment instances are created by
// EnvironmentManager;
template <typename ValueT>
class EnvironmentManager {
 public:
  EnvironmentManager() {}
  ~EnvironmentManager() { ClearAllFrames(); }

  std::shared_ptr<Environment<ValueT>> New(
      const std::shared_ptr<Environment<ValueT>>& env) {
    auto ptr = Environment<ValueT>::New(env);
    weak_envs_.push_back(ptr);
    return ptr;
  }

  std::shared_ptr<Environment<ValueT>> NewInitEnv(const Frame<ValueT>& frame) {
    auto ptr = Environment<ValueT>::NewInitEnv(frame);
    weak_envs_.push_back(ptr);
    return ptr;
  }

  void ClearAllFrames() {
    for (const auto& weak_env : weak_envs_) {
      if (auto env = weak_env.lock()) {
        env->ClearFrame();
      }
    }
    weak_envs_.clear();
  }

 private:
  std::list<std::weak_ptr<Environment<ValueT>>> weak_envs_;
};

}  // namespace pexpr
