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

#include "paddle/pir/include/dialect/pexpr/adt.h"
#include "paddle/pir/include/dialect/pexpr/core_expr.h"
#include "paddle/pir/include/dialect/pexpr/error.h"
#include "paddle/pir/include/dialect/pexpr/value.h"

namespace pexpr {

template <typename ValueT>
class CoreExprInterpreter {
 public:
  using EnvMgr = EnvironmentManager<ValueT>;
  using Env = Environment<ValueT>;
  explicit CoreExprInterpreter(EnvMgr* env_mgr,
                               const Frame<ValueT>& builtin_frame)
      : env_mgr_(env_mgr), builtin_frame_(builtin_frame) {}
  CoreExprInterpreter(const CoreExprInterpreter&) = delete;
  CoreExprInterpreter(CoreExprInterpreter&&) = delete;

  Result<ValueT> operator()(const Closure<ValueT>& closure,
                            const std::vector<ValueT>& args) {
    return InterpretClosure(closure, args);
  }

  const Frame<ValueT>& builtin_frame() const { return builtin_frame_; }

  Result<ValueT> Interpret(const CoreExpr& code,
                           const std::shared_ptr<Env>& env) {
    return code.Match(
        [&](const Atomic<CoreExpr>& atomic) {
          return InterpretAtomic(atomic, env);
        },
        [&](const ComposedCallAtomic<CoreExpr>& composed_call) {
          return InterpretComposedCall(composed_call, env);
        });
  }

  Result<ValueT> InterpretAtomic(const Atomic<CoreExpr>& atomic,
                                 const std::shared_ptr<Env>& env) {
    return atomic.Match(
        [&](const Lambda<CoreExpr>& lambda) -> Result<ValueT> {
          return NaiveClosure<ValueT>{lambda, env};
        },
        [&](const tVar<std::string>& var) -> Result<ValueT> {
          return env->Get(var.value())
              .Match(
                  [&](const Error& error) -> Result<ValueT> {
                    return NameError{std::string("name '") + var.value() +
                                     "' is not defined."};
                  },
                  [&](const auto& val) -> Result<ValueT> { return val; });
        },
        [&](const auto& val) -> Result<ValueT> { return ValueT{val}; });
  }

  Result<ValueT> InterpretComposedCall(
      const ComposedCallAtomic<CoreExpr>& composed_call,
      const std::shared_ptr<Env>& env) {
    Result<ValueT> inner_func = InterpretAtomic(composed_call->inner_func, env);
    if (inner_func.Has<Error>()) {
      return inner_func.Get<Error>();
    }
    std::vector<ValueT> arg_values;
    arg_values.reserve(composed_call->args.size());
    for (const auto& arg : composed_call->args) {
      Result<ValueT> arg_value = InterpretAtomic(arg, env);
      if (arg_value.Has<Error>()) {
        return arg_value.Get<Error>();
      }
      arg_values.push_back(arg_value.Get<ValueT>());
    }
    Result<ValueT> inner_ret =
        InterpretCall(inner_func.Get<ValueT>(), arg_values);
    if (inner_ret.Has<Error>()) {
      return inner_ret.Get<Error>();
    }
    return composed_call->outter_func.Match(
        [&](const Lambda<CoreExpr>& lambda) -> Result<ValueT> {
          return InterpretLambda(lambda, env, {inner_ret.Get<ValueT>()});
        },
        [&](const auto&) -> Result<ValueT> {
          Result<ValueT> outter_func =
              InterpretAtomic(composed_call->outter_func, env);
          if (outter_func.Has<Error>()) {
            return outter_func.Get<Error>();
          }
          return InterpretCall(outter_func.Get<ValueT>(),
                               {inner_ret.Get<ValueT>()});
        });
  }

  Result<ValueT> InterpretClosure(const Closure<ValueT>& closure,
                                  const std::vector<ValueT>& args) {
    return closure.Match(
        [&](const NaiveClosure& impl) {
          return InterpretNaiveClosure(impl, args);
        },
        [&](const MethodClosure& impl) {
          return InterpretMethodClosure(impl, args);
        });
  }

  Result<ValueT> InterpretNaiveClosure(const NaiveClosure<ValueT>& closure,
                                       const std::vector<ValueT>& args) {
    const auto& new_env = env_mgr_->New(closure->environment);
    return InterpretLambda(closure->lambda, new_env, args);
  }

  Result<ValueT> InterpretLambda(const Lambda<CoreExpr>& lambda,
                                 const std::shared_ptr<Env>& env,
                                 const std::vector<ValueT>& args) {
    if (args.size() != lambda->args.size()) {
      return TypeError{std::string("<lambda>() takes ") +
                       std::to_string(lambda->args.size()) +
                       " positional arguments but " +
                       std::to_string(args.size()) + " was given"};
    }
    for (int i = 0; i < args.size(); ++i) {
      const auto& arg_name = lambda->args.at(i).value();
      if (!env->Set(arg_name, args.at(i))) {
        return SyntaxError{"duplicate argument '" + arg_name +
                           "' in function definition"};
      }
    }
    return Interpret(lambda->body, env);
  }

  Result<ValueT> InterpretMethodClosure(const MethodClosure<ValueT>& method,
                                        const std::vector<ValueT>& args) {
    std::vector<ValueT> new_args;
    new_args.reserve(args.size() + 1);
    new_args.push_back(method.obj);
    new_args.insert(new_args.end(), args.begin(), args.end());
    return InterpretCall(method->func, new_args);
  }

  Result<ValueT> InterpretCall(const ValueT& f,
                               const std::vector<ValueT>& args) {
    return f.Match(
        [&](const BuiltinFuncType<ValueT>& func) -> Result<ValueT> {
          const auto& Func =
              [this](const Closure<ValueT>& closure,
                     const std::vector<ValueT>& args) -> Result<ValueT> {
            return InterpretClosure(closure, args);
          };
          return func(Func, args);
        },
        [&](const Closure<ValueT>& closure) -> Result<ValueT> {
          return InterpretClosure(closure, args);
        },
        [&](const auto& other) -> Result<ValueT> {
          return TypeError{std::string("'") + GetBuiltinTypeName(f) +
                           "' object is not callable"};
        });
  }

 protected:
  EnvMgr* env_mgr_;
  Frame<ValueT> builtin_frame_;
};

}  // namespace pexpr
