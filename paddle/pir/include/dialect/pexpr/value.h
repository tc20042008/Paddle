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
#include "paddle/pir/include/dialect/pexpr/atomic.h"
#include "paddle/pir/include/dialect/pexpr/core_expr.h"
#include "paddle/pir/include/dialect/pexpr/error.h"

namespace pexpr {

template <typename CustomT>
class Value;

template <typename ValueT>
struct ObjectImpl {
  std::unordered_map<std::string, ValueT> storage;

  size_t size() const { return storage.size(); }

  void clear() { storage.clear(); }

  Result<ValueT> Get(const std::string& var) const {
    const auto& iter = storage.find(var);
    if (iter == storage.end()) {
      return AttributeError{"object has no attribute '" + var + "'"};
    }
    return iter->second;
  }

  bool Set(const std::string& var, const ValueT& val) {
    return storage.insert({var, val}).second;
  }
};
template <typename ValueT>
DEFINE_ADT_RC(Object, ObjectImpl<ValueT>);

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

  void Destory() { frame_obj->clear(); }

  Object<ValueT> frame_obj;
};

template <typename ValueT>
struct EnvironmentManager;

template <typename ValueT>
class Environment {
 public:
  Result<ValueT> Get(const std::string& var) const {
    Result<ValueT> res = builtin_frame_.Get(var);
    if (res.template Has<ValueT>()) {
      return res;
    }
    res = frame_.Get(var);
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

  void Destory() {
    parent_ = nullptr;
    frame_.Destory();
  }

 private:
  static std::shared_ptr<Environment> New(const Frame<ValueT>& builtin_frame) {
    return std::shared_ptr<Environment>(new Environment(builtin_frame));
  }

  static std::shared_ptr<Environment> New(
      const std::shared_ptr<Environment>& parent) {
    return std::shared_ptr<Environment>(new Environment(parent));
  }

  explicit Environment(const Frame<ValueT>& builtin_frame)
      : parent_(nullptr), builtin_frame_(builtin_frame), frame_() {}

  explicit Environment(const std::shared_ptr<Environment>& parent)
      : parent_(parent), builtin_frame_(parent->builtin_frame_), frame_() {}

  Environment(const Environment&) = delete;
  Environment(Environment&&) = delete;

  friend class EnvironmentManager<ValueT>;

  std::shared_ptr<Environment> parent_;
  Frame<ValueT> builtin_frame_;
  Frame<ValueT> frame_;
};

template <typename ValueT>
struct ClosureImpl {
  Lambda<CoreExpr> lambda;
  std::shared_ptr<Environment<ValueT>> environment;
};

template <typename ValueT>
DEFINE_ADT_RC(Closure, const ClosureImpl<ValueT>);

template <typename ValueT>
using InterpretFuncType = std::function<Result<ValueT>(
    const Closure<ValueT>& closure, const std::vector<ValueT>& args)>;

template <typename ValueT>
using BuiltinFuncType =
    std::function<Result<ValueT>(const InterpretFuncType<ValueT>& Interpret,
                                 const std::vector<ValueT>& args)>;

template <typename CustomT>
using ValueBase = std::variant<CustomT,
                               Nothing,
                               int64_t,
                               std::string,
                               bool,
                               Closure<Value<CustomT>>,
                               adt::List<Value<CustomT>>,
                               Object<Value<CustomT>>,
                               BuiltinFuncType<Value<CustomT>>>;

template <typename CustomT>
struct Value : public ValueBase<CustomT> {
  using ValueBase<CustomT>::ValueBase;
  DEFINE_ADT_VARIANT_METHODS(ValueBase<CustomT>);
};

template <typename CustomT>
const char* GetBuiltinTypeName(const Value<CustomT>& val) {
  using ValueT = Value<CustomT>;
  return val.Match(
      [](const bool c) -> const char* { return "bool"; },
      [](const int64_t c) -> const char* { return "int"; },
      [](const std::string& c) -> const char* { return "str"; },
      [](const Nothing&) -> const char* { return "None"; },
      [](const adt::List<ValueT>& list) -> const char* { return "list"; },
      [](const Object<ValueT>& obj) -> const char* { return "object"; },
      [](const Closure<ValueT>& closure) -> const char* { return "closure"; },
      [](const BuiltinFuncType<ValueT>& closure) -> const char* {
        return "builtin_function";
      },
      [](const CustomT&) -> const char* { return "custom_type"; });
}

template <typename CustomT>
Result<Value<CustomT>> CustomGetAttr(const CustomT& val,
                                     const std::string& name);

template <typename CustomT>
Result<Value<CustomT>> ObjectGetAttr(const Value<CustomT>& val,
                                     const std::string& name) {
  using ValueT = Value<CustomT>;
  return val.Match(
      [&](const Object<ValueT>& obj) -> Result<ValueT> {
        const auto& iter = obj.find(name);
        if (iter == obj.end()) {
          return AttributeError{std::string("no attribute '") + name +
                                "' found."};
        }
        return iter->second;
      },
      [&](const CustomT& custom_val) -> Result<ValueT> {
        return CustomGetAttr(custom_val, name);
      },
      [&](const auto& other) -> Result<ValueT> {
        return AttributeError{std::string("no attribute '") + name +
                              "' found."};
      });
}

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
  ~EnvironmentManager() {
    for (const auto& weak_env : weak_envs_) {
      if (auto env = weak_env.lock()) {
        env->Destory();
      }
    }
  }

  template <typename T>
  std::shared_ptr<Environment<ValueT>> New(T&& arg) {
    auto ptr = Environment<ValueT>::New(std::forward<T>(arg));
    weak_envs_.push_back(ptr);
    return ptr;
  }

 private:
  std::list<std::weak_ptr<Environment<ValueT>>> weak_envs_;
};

}  // namespace pexpr
