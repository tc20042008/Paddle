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

#include <glog/logging.h>
#include "paddle/pir/include/dialect/pexpr/adt.h"
#include "paddle/pir/include/dialect/pexpr/builtin_functions.h"
#include "paddle/pir/include/dialect/pexpr/core_expr.h"
#include "paddle/pir/include/dialect/pexpr/error.h"
#include "paddle/pir/include/dialect/pexpr/value.h"
#include "paddle/pir/include/dialect/pexpr/value_method_class.h"

namespace pexpr {

template <typename ValueT>
class CpsExprInterpreter : public CpsInterpreterBase<ValueT> {
 public:
  using EnvMgr = EnvironmentManager<ValueT>;
  using Env = Environment<ValueT>;
  CpsExprInterpreter() : env_mgr_(new EnvMgr()), builtin_env_() {}
  CpsExprInterpreter(const std::shared_ptr<EnvMgr>& env_mgr,
                     const Frame<ValueT>& frame)
      : env_mgr_(env_mgr), builtin_env_(env_mgr->NewInitEnv(frame)) {}
  CpsExprInterpreter(const CpsExprInterpreter&) = delete;
  CpsExprInterpreter(CpsExprInterpreter&&) = delete;

  const std::shared_ptr<EnvMgr>& env_mgr() const { return env_mgr_; }
  const std::shared_ptr<Env>& builtin_env() const { return builtin_env_; }

  Result<ValueT> Interpret(const Lambda<CoreExpr>& lambda,
                           const std::vector<ValueT>& args) {
    Closure<ValueT> closure{lambda, env_mgr()->New(builtin_env())};
    const auto& ret = Interpret(closure, args);
    env_mgr()->ClearAllFrames();
    return ret;
  }

  Result<ValueT> Interpret(const Closure<ValueT>& closure,
                           const std::vector<ValueT>& args) override {
    ComposedCallImpl<ValueT> composed_call{&BuiltinHalt<ValueT>, closure, args};
    const auto& ret = InterpretComposedCallUntilHalt(&composed_call);
    ADT_RETURN_IF_ERROR(ret);
    if (!IsHalt(composed_call.inner_func)) {
      return RuntimeError{"CpsExprInterpreter does not halt."};
    }
    if (composed_call.args.size() != 1) {
      return RuntimeError{
          std::string() + "halt function takes 1 argument. but " +
          std::to_string(composed_call.args.size()) + " were given."};
    }
    return composed_call.args.at(0);
  }

 protected:
  Result<adt::Ok> InterpretComposedCallUntilHalt(
      ComposedCallImpl<ValueT>* composed_call) {
    while (!IsHalt(composed_call->inner_func)) {
      const auto& ret = InterpretComposedCall(composed_call);
      ADT_RETURN_IF_ERROR(ret);
    }
    return adt::Ok{};
  }

  Result<adt::Ok> InterpretComposedCall(
      ComposedCallImpl<ValueT>* composed_call) {
    return composed_call->inner_func.Match(
        [&](const BuiltinFuncType<ValueT>& func) -> Result<adt::Ok> {
          return InterpretBuiltinFuncCall(func, composed_call);
        },
        [&](const CpsBuiltinHighOrderFuncType<ValueT>& func)
            -> Result<adt::Ok> { return func(this, composed_call); },
        [&](const Method<ValueT>& method) -> Result<adt::Ok> {
          return method->func.Match(
              [&](const BuiltinFuncType<ValueT>& func) {
                return InterpretBuiltinMethodCall(
                    func, method->obj, composed_call);
              },
              [&](const auto&) {
                return InterpretMethodCall(method, composed_call);
              });
        },
        [&](const Closure<ValueT>& closure) -> Result<adt::Ok> {
          return InterpretClosureCall(composed_call->outter_func,
                                      closure,
                                      composed_call->args,
                                      composed_call);
        },
        [&](const builtin_symbol::Symbol& symbol) -> Result<adt::Ok> {
          return InterpretBuiltinSymbolCall(symbol, composed_call);
        },
        [&](const auto& other) -> Result<adt::Ok> {
          return TypeError{
              std::string("'") +
              MethodClass<ValueT>::Name(composed_call->inner_func) +
              "' object is not callable"};
        });
  }

  bool IsHalt(const ValueT& func) {
    return func.Match(
        [&](BuiltinFuncType<ValueT> f) { return f == &BuiltinHalt<ValueT>; },
        [&](const auto&) { return false; });
  }

  Result<ValueT> InterpretAtomic(const std::shared_ptr<Env>& env,
                                 const Atomic<CoreExpr>& atomic) {
    return atomic.Match(
        [&](const Lambda<CoreExpr>& lambda) -> Result<ValueT> {
          return Closure<ValueT>{lambda, env};
        },
        [&](const Symbol& symbol) -> Result<ValueT> {
          return symbol.Match(
              [&](const tVar<std::string>& var) -> Result<ValueT> {
                return env->Get(var.value())
                    .Match(
                        [&](const Error& error) -> Result<ValueT> {
                          return NameError{std::string("name '") + var.value() +
                                           "' is not defined."};
                        },
                        [&](const auto& val) -> Result<ValueT> { return val; });
              },
              [&](const builtin_symbol::Symbol& symbol) -> Result<ValueT> {
                return symbol.Match(
                    [&](const builtin_symbol::Nothing&) -> Result<ValueT> {
                      return adt::Nothing{};
                    },
                    [&](const auto&) -> Result<ValueT> { return symbol; });
              });
        },
        [&](int64_t c) -> Result<ValueT> { return ArithmeticValue{c}; },
        [&](bool c) -> Result<ValueT> { return ArithmeticValue{c}; },
        [&](const std::string& val) -> Result<ValueT> { return ValueT{val}; });
  }

  Result<adt::Ok> InterpretBuiltinSymbolCall(
      const builtin_symbol::Symbol& symbol,
      ComposedCallImpl<ValueT>* ret_composed_call) {
    return symbol.Match(
        [&](const builtin_symbol::If&) -> Result<adt::Ok> {
          ret_composed_call->inner_func = &CpsBuiltinIf<ValueT>;
          return adt::Ok{};
        },
        [&](const builtin_symbol::Apply&) -> Result<adt::Ok> {
          ret_composed_call->inner_func = &CpsBuiltinApply<ValueT>;
          return adt::Ok{};
        },
        [&](const builtin_symbol::Nothing&) -> Result<adt::Ok> {
          return TypeError{"'None' is not callable"};
        },
        [&](const builtin_symbol::Id&) -> Result<adt::Ok> {
          ret_composed_call->inner_func = &BuiltinIdentity<ValueT>;
          return adt::Ok{};
        },
        [&](const builtin_symbol::List&) -> Result<adt::Ok> {
          ret_composed_call->inner_func = &BuiltinList<ValueT>;
          return adt::Ok{};
        },
        [&](const builtin_symbol::Op& op) -> Result<adt::Ok> {
          return op.Match([&](auto impl) -> Result<adt::Ok> {
            using BuiltinSymbol = decltype(impl);
            if constexpr (BuiltinSymbol::num_operands == 1) {
              return this
                  ->template InterpretBuiltinUnarySymbolCall<BuiltinSymbol>(
                      ret_composed_call);
            } else if constexpr (BuiltinSymbol::num_operands == 2) {
              return this
                  ->template InterpretBuiltinBinarySymbolCall<BuiltinSymbol>(
                      ret_composed_call);
            } else {
              static_assert(true, "NotImplemented");
              return RuntimeError{"NotImplemented."};
            }
          });
        });
  }

  template <typename BuiltinSymbol>
  Result<adt::Ok> InterpretBuiltinUnarySymbolCall(
      ComposedCallImpl<ValueT>* ret_composed_call) {
    if (ret_composed_call->args.size() != 1) {
      return TypeError{std::string() + "'" + BuiltinSymbol::Name() +
                       "' takes 1 argument. but " +
                       std::to_string(ret_composed_call->args.size()) +
                       " were given."};
    }
    const auto& operand = ret_composed_call->args.at(0);
    const auto& opt_func =
        MethodClass<ValueT>::template GetBuiltinUnaryFunc<BuiltinSymbol>(
            operand);
    if (!opt_func.has_value()) {
      return TypeError{std::string() + "unsupported operand type for " +
                       GetBuiltinSymbolDebugString<BuiltinSymbol>() + ": '" +
                       MethodClass<ValueT>::Name(operand) + "'"};
    }
    const auto& opt_ret = opt_func.value()(operand);
    ADT_RETURN_IF_ERROR(opt_ret);
    const auto& ret = opt_ret.GetOkValue();
    ret_composed_call->args = {ret};
    ret_composed_call->inner_func = ret_composed_call->outter_func;
    ret_composed_call->outter_func = &BuiltinHalt<ValueT>;
    return adt::Ok{};
  }

  template <typename BuiltinSymbol>
  Result<adt::Ok> InterpretBuiltinBinarySymbolCall(
      ComposedCallImpl<ValueT>* ret_composed_call) {
    if (ret_composed_call->args.size() != 2) {
      return TypeError{std::string() + "'" + BuiltinSymbol::Name() +
                       "' takes 2 argument. but " +
                       std::to_string(ret_composed_call->args.size()) +
                       " were given."};
    }
    const auto& lhs = ret_composed_call->args.at(0);
    const auto& opt_func =
        MethodClass<ValueT>::template GetBuiltinBinaryFunc<BuiltinSymbol>(lhs);
    if (!opt_func.has_value()) {
      return TypeError{std::string() + "unsupported operand type for " +
                       GetBuiltinSymbolDebugString<BuiltinSymbol>() + ": '" +
                       MethodClass<ValueT>::Name(lhs) + "'"};
    }
    const auto& rhs = ret_composed_call->args.at(1);
    const auto& opt_ret = opt_func.value()(lhs, rhs);
    ADT_RETURN_IF_ERROR(opt_ret);
    const auto& ret = opt_ret.GetOkValue();
    ret_composed_call->args = {ret};
    ret_composed_call->inner_func = ret_composed_call->outter_func;
    ret_composed_call->outter_func = &BuiltinHalt<ValueT>;
    return adt::Ok{};
  }

  Result<adt::Ok> InterpretClosureCall(
      const ValueT& continuation,
      const Closure<ValueT>& closure,
      const std::vector<ValueT>& args,
      ComposedCallImpl<ValueT>* ret_composed_call) {
    const auto& new_env = env_mgr_->New(closure->environment);
    new_env->Set(kBuiltinReturn(), continuation);
    return InterpretLambdaCall(
        new_env, continuation, closure->lambda, args, ret_composed_call);
  }

  Result<adt::Ok> InterpretLambdaCall(
      const std::shared_ptr<Env>& env,
      const ValueT& outter_func,
      const Lambda<CoreExpr>& lambda,
      const std::vector<ValueT>& args,
      ComposedCallImpl<ValueT>* ret_composed_call) override {
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
    return lambda->body.Match(
        [&](const Atomic<CoreExpr>& atomic) -> Result<adt::Ok> {
          const auto& val = InterpretAtomic(env, atomic);
          ADT_RETURN_IF_ERROR(val);
          ret_composed_call->outter_func = outter_func;
          ret_composed_call->inner_func = &BuiltinIdentity<ValueT>;
          ret_composed_call->args = {val.GetOkValue()};
          return adt::Ok{};
        },
        [&](const ComposedCallAtomic<CoreExpr>& core_expr) -> Result<adt::Ok> {
          return InterpretComposedCallAtomic(env, core_expr, ret_composed_call);
        });
  }

  Result<adt::Ok> InterpretComposedCallAtomic(
      const std::shared_ptr<Env>& env,
      const ComposedCallAtomic<CoreExpr>& core_expr,
      ComposedCallImpl<ValueT>* ret_composed_call) {
    const auto& new_outter_func = InterpretAtomic(env, core_expr->outter_func);
    ADT_RETURN_IF_ERROR(new_outter_func);
    const auto& new_inner_func = InterpretAtomic(env, core_expr->inner_func);
    ADT_RETURN_IF_ERROR(new_inner_func);
    std::vector<ValueT> args;
    args.reserve(core_expr->args.size());
    for (const auto& arg_expr : core_expr->args) {
      const auto& arg = InterpretAtomic(env, arg_expr);
      ADT_RETURN_IF_ERROR(arg);
      args.emplace_back(arg.GetOkValue());
    }
    ret_composed_call->outter_func = new_outter_func.GetOkValue();
    ret_composed_call->inner_func = new_inner_func.GetOkValue();
    ret_composed_call->args = std::move(args);
    return adt::Ok{};
  }

  Result<adt::Ok> InterpretBuiltinFuncCall(
      const BuiltinFuncType<ValueT>& func,
      ComposedCallImpl<ValueT>* composed_call) {
    return InterpretBuiltinMethodCall(
        func, ValueT{adt::Nothing{}}, composed_call);
  }

  Result<adt::Ok> InterpretBuiltinMethodCall(
      const BuiltinFuncType<ValueT>& func,
      const ValueT& obj,
      ComposedCallImpl<ValueT>* composed_call) {
    const auto original_outter_func = composed_call->outter_func;
    const auto& opt_inner_ret = func(obj, composed_call->args);
    ADT_RETURN_IF_ERROR(opt_inner_ret);
    const auto& inner_ret = opt_inner_ret.GetOkValue();
    if (original_outter_func.template Has<Closure<ValueT>>()) {
      const auto& closure =
          original_outter_func.template Get<Closure<ValueT>>();
      return InterpretLambdaCall(closure->environment,
                                 ValueT{&BuiltinHalt<ValueT>},
                                 closure->lambda,
                                 {inner_ret},
                                 composed_call);
    }
    composed_call->outter_func = &BuiltinHalt<ValueT>;
    composed_call->inner_func = original_outter_func;
    composed_call->args = {inner_ret};
    return adt::Ok{};
  }
  Result<adt::Ok> InterpretMethodCall(const Method<ValueT>& method,
                                      ComposedCallImpl<ValueT>* composed_call) {
    std::vector<ValueT> new_args;
    new_args.reserve(composed_call->args.size() + 1);
    new_args.emplace_back(method->obj);
    for (const auto& arg : composed_call->args) {
      new_args.emplace_back(arg);
    }
    composed_call->inner_func = method->func;
    composed_call->args = std::move(new_args);
    return adt::Ok{};
  }

  std::shared_ptr<EnvMgr> env_mgr_;
  std::shared_ptr<Env> builtin_env_;

 private:
};

}  // namespace pexpr
