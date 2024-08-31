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

using adt::Nothing;

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

  bool operator==(const ObjectImpl& other) const { return &other == this; }
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

  void ClearFrame() { frame_obj->clear(); }

  Object<ValueT> frame_obj;
};

template <typename ValueT>
struct EnvironmentManager;

template <typename ValueT>
class Environment {
 public:
  Result<ValueT> Get(const std::string& var) const {
    const Result<ValueT>& builtin_val = builtin_frame_.Get(var);
    if (builtin_val.template Has<ValueT>()) {
      return builtin_val;
    }
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
struct NaiveClosureImpl {
  Lambda<CoreExpr> lambda;
  std::shared_ptr<Environment<ValueT>> environment;

  bool operator==(const NaiveClosureImpl& other) const {
    return other.lambda == this->lambda &&
           other.environment == this->environment;
  }
};

template <typename ValueT>
DEFINE_ADT_RC(NaiveClosure, const NaiveClosureImpl<ValueT>);

template <typename ValueT>
struct MethodClosureImpl {
  ValueT obj;
  ValueT func;

  bool operator==(const MethodClosureImpl& other) const {
    return other.obj == this->obj && other.func == this->func;
  }
};

template <typename ValueT>
DEFINE_ADT_RC(MethodClosure, const MethodClosureImpl<ValueT>);

template <typename ValueT>
using ClosureImpl = std::variant<NaiveClosure<ValueT>, MethodClosure<ValueT>>;

template <typename ValueT>
struct Closure : public ClosureImpl<ValueT> {
  using ClosureImpl<ValueT>::ClosureImpl;
  DEFINE_ADT_VARIANT_METHODS(ClosureImpl<ValueT>);
};

template <typename ValueT>
using InterpretFuncType = std::function<Result<ValueT>(
    const Closure<ValueT>& closure, const std::vector<ValueT>& args)>;

template <typename ValueT>
using BuiltinFuncType =
    Result<ValueT> (*)(const InterpretFuncType<ValueT>& Interpret,
                       const std::vector<ValueT>& args);

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

template <typename T>
struct GetBuiltinTypeNameImplHelper;

template <>
struct GetBuiltinTypeNameImplHelper<bool> {
  static const char* Call() { return "bool"; }
};

template <>
struct GetBuiltinTypeNameImplHelper<int64_t> {
  static const char* Call() { return "int"; }
};

template <>
struct GetBuiltinTypeNameImplHelper<std::string> {
  static const char* Call() { return "str"; }
};

template <>
struct GetBuiltinTypeNameImplHelper<Nothing> {
  static const char* Call() { return "NoneType"; }
};

template <typename ValueT>
struct GetBuiltinTypeNameImplHelper<adt::List<ValueT>> {
  static const char* Call() { return "list"; }
};

template <typename ValueT>
struct GetBuiltinTypeNameImplHelper<Object<ValueT>> {
  static const char* Call() { return "object"; }
};

template <typename ValueT>
struct GetBuiltinTypeNameImplHelper<Closure<ValueT>> {
  static const char* Call() { return "closure_or_method"; }
};

template <typename ValueT>
struct GetBuiltinTypeNameImplHelper<NaiveClosure<ValueT>> {
  static const char* Call() { return "closure"; }
};

template <typename ValueT>
struct GetBuiltinTypeNameImplHelper<MethodClosure<ValueT>> {
  static const char* Call() { return "method"; }
};

template <typename ValueT>
struct GetBuiltinTypeNameImplHelper<BuiltinFuncType<ValueT>> {
  static const char* Call() { return "builtin_function"; }
};

template <typename T>
const char* GetBuiltinTypeNameImpl() {
  return GetBuiltinTypeNameImplHelper<T>::Call();
}

template <typename CustomT>
const char* GetBuiltinTypeName(const Value<CustomT>& val) {
  using ValueT = Value<CustomT>;
  return val.Match(
      [](const Closure<ValueT>& closure) -> const char* {
        return closure.Match([&](const auto& impl) -> const char* {
          return GetBuiltinTypeNameImpl<std::decay_t<decltype(impl)>>();
        });
      },
      [](const CustomT&) -> const char* { return "custom_type"; },
      [](const auto& impl) -> const char* {
        return GetBuiltinTypeNameImpl<std::decay_t<decltype(impl)>>();
      });
}

template <typename T, typename ValueT>
struct CastToBuiltinValueHelpr {
  static Result<T> Call(const ValueT& value) {
    if (!value.template Has<T>()) {
      return TypeError{std::string() + "cast failed. expected type: " +
                       GetBuiltinTypeNameImpl<T>() +
                       ", actual type: " + GetBuiltinTypeName(value)};
    }
    return value.template Get<T>();
  }
};

template <typename ValueT>
struct CastToBuiltinValueHelpr<NaiveClosure<ValueT>, ValueT> {
  static Result<NaiveClosure<ValueT>> Call(const ValueT& value) {
    const auto& opt_closure =
        CastToBuiltinValueHelpr<Closure<ValueT>, ValueT>::Call(value);
    ADT_RETURN_IF_ERROR(opt_closure);
    const auto& closure = opt_closure.GetOkValue();
    if (!closure.template Has<NaiveClosure<ValueT>>()) {
      return TypeError{
          std::string() +
          "cast failed. expected type: 'naive_closure', actual type: " +
          GetBuiltinTypeName(value)};
    }
    return closure.template Get<NaiveClosure<ValueT>>();
  }
};

template <typename ValueT>
struct CastToBuiltinValueHelpr<MethodClosure<ValueT>, ValueT> {
  static Result<MethodClosure<ValueT>> Call(const ValueT& value) {
    const auto& opt_closure =
        CastToBuiltinValueHelpr<Closure<ValueT>, ValueT>::Call(value);
    ADT_RETURN_IF_ERROR(opt_closure);
    const auto& closure = opt_closure.GetOkValue();
    if (!closure.template Has<MethodClosure<ValueT>>()) {
      return TypeError{
          std::string() +
          "cast failed. expected type: 'method_closure', actual type: " +
          GetBuiltinTypeName(value)};
    }
    return closure.template Get<MethodClosure<ValueT>>();
  }
};

template <typename T, typename ValueT>
Result<T> CastToBuiltinValue(const ValueT& value) {
  return CastToBuiltinValueHelpr<T, ValueT>::Call(value);
}

template <typename CustomT>
Result<Value<CustomT>> CustomGetAttr(const CustomT& val,
                                     const std::string& name);

template <typename CustomT>
Result<Value<CustomT>> ValueGetAttr(const Value<CustomT>& val,
                                    const std::string& name) {
  using ValueT = Value<CustomT>;
  return val.Match(
      [&](const Object<ValueT>& obj) -> Result<ValueT> {
        const auto& iter = obj->storage.find(name);
        if (iter == obj->storage.end()) {
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

template <typename CustomT>
Result<Value<CustomT>> CustomGetItem(const CustomT& val,
                                     const Value<CustomT>& idx);

template <typename CustomT>
Result<Value<CustomT>> ValueGetItem(const Value<CustomT>& val,
                                    const Value<CustomT>& idx) {
  using ValueT = Value<CustomT>;
  return val.Match(
      [&](const adt::List<ValueT>& obj) -> Result<ValueT> {
        return idx.Match(
            [&](int64_t idx) -> Result<ValueT> {
              if (idx < 0) {
                idx += obj->size();
              }
              if (idx >= 0 && idx < obj->size()) {
                return obj->at(idx);
              }
              return IndexError{"list index out of range"};
            },
            [&](const auto&) -> Result<ValueT> {
              return TypeError{std::string() +
                               "list indices must be integers, not " +
                               GetBuiltinTypeName(idx)};
            });
      },
      [&](const CustomT& custom_val) -> Result<ValueT> {
        return CustomGetItem(custom_val, idx);
      },
      [&](const auto& other) -> Result<ValueT> {
        return TypeError{std::string() + "'" + GetBuiltinTypeName(val) +
                         "' object is not subscriptable"};
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
  ~EnvironmentManager() { ClearAllFrames(); }

  template <typename T>
  std::shared_ptr<Environment<ValueT>> New(T&& arg) {
    auto ptr = Environment<ValueT>::New(std::forward<T>(arg));
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
