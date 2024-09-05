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
#include "paddle/pir/include/dialect/pexpr/arithmetic_type.h"
#include "paddle/pir/include/dialect/pexpr/arithmetic_value.h"
#include "paddle/pir/include/dialect/pexpr/atomic.h"
#include "paddle/pir/include/dialect/pexpr/core_expr.h"
#include "paddle/pir/include/dialect/pexpr/error.h"
#include "paddle/pir/include/dialect/pexpr/pointer_type.h"
#include "paddle/pir/include/dialect/pexpr/pointer_value.h"

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

template <typename ValueT>
struct ClosureImpl {
  Lambda<CoreExpr> lambda;
  std::shared_ptr<Environment<ValueT>> environment;

  bool operator==(const ClosureImpl& other) const {
    return other.lambda == this->lambda &&
           other.environment == this->environment;
  }
};

template <typename ValueT>
DEFINE_ADT_RC(Closure, const ClosureImpl<ValueT>);

template <typename ValueT>
struct MethodImpl {
  ValueT obj;
  ValueT func;

  bool operator==(const MethodImpl& other) const {
    return other.obj == this->obj && other.func == this->func;
  }
};

template <typename ValueT>
DEFINE_ADT_RC(Method, const MethodImpl<ValueT>);

template <typename ValueT>
using InterpretFuncType = std::function<Result<ValueT>(
    const ValueT& f, const std::vector<ValueT>& args)>;

template <typename ValueT>
class CpsInterpreterBase {
 public:
  virtual Result<adt::Ok> InterpretLambdaCall(
      const std::shared_ptr<Environment<ValueT>>& env,
      const ValueT& outter_func,
      const Lambda<CoreExpr>& lambda,
      const std::vector<ValueT>& args,
      ComposedCallImpl<ValueT>* ret_composed_call) = 0;
};

template <typename ValueT>
using BuiltinFuncType = Result<ValueT> (*)(const ValueT&,
                                           const std::vector<ValueT>& args);

template <typename ValueT>
using CpsBuiltinHighOrderFuncType =
    Result<adt::Ok> (*)(CpsInterpreterBase<ValueT>* CpsInterpret,
                        ComposedCallImpl<ValueT>* composed_call);

template <typename CustomT>
using ValueBase = std::variant<CustomT,
                               Nothing,
                               ArithmeticType,
                               PointerType,
                               ArithmeticValue,
                               PointerValue,
                               std::string,
                               Closure<Value<CustomT>>,
                               Method<Value<CustomT>>,
                               adt::List<Value<CustomT>>,
                               Object<Value<CustomT>>,
                               BuiltinFuncType<Value<CustomT>>,
                               CpsBuiltinHighOrderFuncType<Value<CustomT>>>;

template <typename CustomT>
struct Value : public ValueBase<CustomT> {
  using ValueBase<CustomT>::ValueBase;
  DEFINE_ADT_VARIANT_METHODS(ValueBase<CustomT>);
};

template <typename T>
struct GetBuiltinTypeNameImplHelper {
  static const char* Call() { return "custom_type"; }
};

template <>
struct GetBuiltinTypeNameImplHelper<ArithmeticType> {
  static const char* Call() { return "arithmetic_type"; }
};

template <>
struct GetBuiltinTypeNameImplHelper<PointerType> {
  static const char* Call() { return "pointer_type"; }
};

template <>
struct GetBuiltinTypeNameImplHelper<ArithmeticValue> {
  static const char* Call() { return "arithmetic_value"; }
};

template <>
struct GetBuiltinTypeNameImplHelper<PointerValue> {
  static const char* Call() { return "pointer_value"; }
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
  static const char* Call() { return "closure"; }
};

template <typename ValueT>
struct GetBuiltinTypeNameImplHelper<Method<ValueT>> {
  static const char* Call() { return "method"; }
};

template <typename ValueT>
struct GetBuiltinTypeNameImplHelper<BuiltinFuncType<ValueT>> {
  static const char* Call() { return "builtin_function"; }
};

template <typename ValueT>
struct GetBuiltinTypeNameImplHelper<CpsBuiltinHighOrderFuncType<ValueT>> {
  static const char* Call() { return "cps_builtin_high_order_function"; }
};

template <typename T>
const char* GetBuiltinTypeNameImpl() {
  return GetBuiltinTypeNameImplHelper<T>::Call();
}

template <typename CustomT>
const char* GetBuiltinTypeName(const Value<CustomT>& val) {
  using ValueT = Value<CustomT>;
  return val.Match(
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

template <typename T, typename ValueT>
Result<T> CastToBuiltinValue(const ValueT& value) {
  return CastToBuiltinValueHelpr<T, ValueT>::Call(value);
}

template <typename T, typename ValueT>
Result<T> CastToArithmeticValue(const ValueT& value) {
  const auto& arithmetic_value = CastToBuiltinValue<ArithmeticValue>(value);
  ADT_RETURN_IF_ERROR(arithmetic_value);
  return arithmetic_value.GetOkValue().template TryGet<T>();
}

template <typename T, typename ValueT>
Result<T> CastToPointerValue(const ValueT& value) {
  const auto& pointer_value = CastToBuiltinValue<PointerValue>(value);
  ADT_RETURN_IF_ERROR(pointer_value);
  return pointer_value.GetOkValue().template TryGet<T>();
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
