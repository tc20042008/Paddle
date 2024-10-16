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
#include <utility>
#include "ap/axpr/adt.h"
#include "ap/axpr/builtin_functions.h"
#include "ap/axpr/core_expr.h"
#include "ap/axpr/error.h"
#include "ap/axpr/value.h"
#include "ap/axpr/value_method_class.h"

namespace ap::axpr {

template <typename ValueT>
class CpsExprInterpreter : public CpsInterpreterBase<ValueT> {
 public:
  using This = CpsExprInterpreter;
  using EnvMgr = EnvironmentManager<ValueT>;
  using Env = Environment<ValueT>;
  CpsExprInterpreter(const std::shared_ptr<EnvMgr>& env_mgr,
                     const Frame<ValueT>& frame)
      : env_mgr_(env_mgr), builtin_env_(env_mgr->NewInitEnv(frame)) {}
  explicit CpsExprInterpreter(const Frame<ValueT>& frame)
      : CpsExprInterpreter(std::make_shared<EnvMgr>(), frame) {}
  CpsExprInterpreter()
      : CpsExprInterpreter(std::make_shared<EnvMgr>(), GetBuiltinFrame()) {}
  CpsExprInterpreter(const CpsExprInterpreter&) = delete;
  CpsExprInterpreter(CpsExprInterpreter&&) = delete;

  const std::shared_ptr<EnvMgr>& env_mgr() const { return env_mgr_; }
  const std::shared_ptr<Env>& builtin_env() const { return builtin_env_; }

  Result<ValueT> Interpret(const Lambda<CoreExpr>& lambda,
                           const std::vector<ValueT>& args) {
    Closure<ValueT> closure{lambda, env_mgr()->New(builtin_env())};
    const auto& ret = Interpret(closure, args);
    return ret;
  }

  Result<ValueT> Interpret(const ValueT& func,
                           const std::vector<ValueT>& args) override {
    ComposedCallImpl<ValueT> composed_call{&BuiltinHalt<ValueT>, func, args};
    ADT_RETURN_IF_ERR(InterpretComposedCallUntilHalt(&composed_call));
    ADT_CHECK(IsHalt(composed_call.inner_func))
        << RuntimeError{"CpsExprInterpreter does not halt."};
    ADT_CHECK(composed_call.args.size() == 1) << RuntimeError{
        std::string() + "halt function takes 1 argument. but " +
        std::to_string(composed_call.args.size()) + " were given."};
    return composed_call.args.at(0);
  }

 protected:
  Result<adt::Ok> InterpretComposedCallUntilHalt(
      ComposedCallImpl<ValueT>* composed_call) {
    while (!IsHalt(composed_call->inner_func)) {
      ADT_RETURN_IF_ERR(InterpretComposedCall(composed_call));
    }
    return adt::Ok{};
  }

  Result<adt::Ok> InterpretComposedCall(
      ComposedCallImpl<ValueT>* composed_call) {
    using TypeT = typename TypeTrait<ValueT>::TypeT;
    return composed_call->inner_func.Match(
        [&](const TypeT& type) -> Result<adt::Ok> {
          return InterpretConstruct(type, composed_call);
        },
        [&](const BuiltinFuncType<ValueT>& func) -> Result<adt::Ok> {
          return InterpretBuiltinFuncCall(func, composed_call);
        },
        [&](const BuiltinHighOrderFuncType<ValueT>& func) -> Result<adt::Ok> {
          return InterpretBuiltinHighOrderFuncCall(func, composed_call);
        },
        [&](const CpsBuiltinHighOrderFuncType<ValueT>& func)
            -> Result<adt::Ok> { return func(this, composed_call); },
        [&](const Method<ValueT>& method) -> Result<adt::Ok> {
          return method->func.Match(
              [&](const BuiltinFuncType<ValueT>& func) {
                return InterpretBuiltinMethodCall(
                    func, method->obj, composed_call);
              },
              [&](const BuiltinHighOrderFuncType<ValueT>& func) {
                return InterpretBuiltinHighOrderMethodCall(
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
        [&](const auto&) -> Result<adt::Ok> {
          const auto& opt_func =
              MethodClass<ValueT>::template GetBuiltinUnaryFunc<
                  builtin_symbol::Call>(composed_call->inner_func);
          ADT_CHECK(opt_func.has_value()) << TypeError{
              std::string("'") +
              MethodClass<ValueT>::Name(composed_call->inner_func) +
              "' object is not callable"};
          ADT_LET_CONST_REF(func, opt_func.value()(composed_call->inner_func));
          composed_call->inner_func = func;
          return adt::Ok{};
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
                ADT_LET_CONST_REF(val, env->Get(var.value()))
                    << adt::errors::NameError{std::string("var '") +
                                              var.value() +
                                              "' is not defined."};
                return val;
              },
              [&](const builtin_symbol::Symbol& symbol) -> Result<ValueT> {
                return symbol;
              });
        },
        [&](adt::Nothing) -> Result<ValueT> { return adt::Nothing{}; },
        [&](bool c) -> Result<ValueT> { return c; },
        [&](int64_t c) -> Result<ValueT> { return c; },
        [&](double c) -> Result<ValueT> { return c; },
        [&](const std::string& val) -> Result<ValueT> { return val; });
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
              return NotImplementedError{"NotImplemented."};
            }
          });
        });
  }

  template <typename BuiltinSymbol>
  Result<adt::Ok> InterpretBuiltinUnarySymbolCall(
      ComposedCallImpl<ValueT>* ret_composed_call) {
    ADT_CHECK(ret_composed_call->args.size() == 1) << TypeError{
        std::string() + "'" + BuiltinSymbol::Name() +
        "' takes 1 argument. but " +
        std::to_string(ret_composed_call->args.size()) + " were given."};
    const auto& operand = ret_composed_call->args.at(0);
    const auto& opt_func =
        MethodClass<ValueT>::template GetBuiltinUnaryFunc<BuiltinSymbol>(
            operand);
    ADT_CHECK(opt_func.has_value())
        << TypeError{std::string() + "unsupported operand type for " +
                     GetBuiltinSymbolDebugString<BuiltinSymbol>() + ": '" +
                     MethodClass<ValueT>::Name(operand) + "'"};
    ADT_LET_CONST_REF(ret, opt_func.value()(operand));
    ret_composed_call->args = {ret};
    ret_composed_call->inner_func = ret_composed_call->outter_func;
    ret_composed_call->outter_func = &BuiltinHalt<ValueT>;
    return adt::Ok{};
  }

  template <typename TypeT>
  Result<adt::Ok> InterpretConstruct(
      const TypeT& type, ComposedCallImpl<ValueT>* ret_composed_call) {
    const auto& opt_func =
        MethodClass<ValueT>::template GetBuiltinUnaryFunc<builtin_symbol::Call>(
            ValueT{type});
    ADT_CHECK(opt_func.has_value()) << TypeError{
        std::string() + "no constructor for type '" + type.Name() + "'"};
    ADT_LET_CONST_REF(constructor, opt_func.value()(ValueT{type}));
    ret_composed_call->inner_func = constructor;
    return adt::Ok{};
  }

  template <typename BuiltinSymbol>
  Result<adt::Ok> InterpretBuiltinBinarySymbolCall(
      ComposedCallImpl<ValueT>* ret_composed_call) {
    ADT_CHECK(ret_composed_call->args.size() == 2) << TypeError{
        std::string() + "'" + BuiltinSymbol::Name() +
        "' takes 2 argument. but " +
        std::to_string(ret_composed_call->args.size()) + " were given."};
    const auto& lhs = ret_composed_call->args.at(0);
    const auto& opt_func =
        MethodClass<ValueT>::template GetBuiltinBinaryFunc<BuiltinSymbol>(lhs);
    ADT_CHECK(opt_func.has_value())
        << TypeError{std::string() + "unsupported operand type for " +
                     GetBuiltinSymbolDebugString<BuiltinSymbol>() + ": '" +
                     MethodClass<ValueT>::Name(lhs) + "'"};
    const auto& rhs = ret_composed_call->args.at(1);
    ADT_LET_CONST_REF(ret, opt_func.value()(lhs, rhs));
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
    ADT_CHECK(args.size() == lambda->args.size()) << TypeError{
        std::string("<lambda>() takes ") + std::to_string(lambda->args.size()) +
        " positional arguments but " + std::to_string(args.size()) +
        " was given"};
    for (int i = 0; i < args.size(); ++i) {
      const auto& arg_name = lambda->args.at(i).value();
      ADT_CHECK(env->Set(arg_name, args.at(i))) << SyntaxError{
          "duplicate argument '" + arg_name + "' in function definition"};
    }
    return lambda->body.Match(
        [&](const Atomic<CoreExpr>& atomic) -> Result<adt::Ok> {
          ADT_LET_CONST_REF(val, InterpretAtomic(env, atomic));
          ret_composed_call->outter_func = outter_func;
          ret_composed_call->inner_func = &BuiltinIdentity<ValueT>;
          ret_composed_call->args = {val};
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
    ADT_LET_CONST_REF(new_outter_func,
                      InterpretAtomic(env, core_expr->outter_func));
    ADT_LET_CONST_REF(new_inner_func,
                      InterpretAtomic(env, core_expr->inner_func));
    std::vector<ValueT> args;
    args.reserve(core_expr->args.size());
    for (const auto& arg_expr : core_expr->args) {
      ADT_LET_CONST_REF(arg, InterpretAtomic(env, arg_expr));
      args.emplace_back(arg);
    }
    ret_composed_call->outter_func = new_outter_func;
    ret_composed_call->inner_func = new_inner_func;
    ret_composed_call->args = std::move(args);
    return adt::Ok{};
  }

  Result<adt::Ok> InterpretBuiltinFuncCall(
      const BuiltinFuncType<ValueT>& func,
      ComposedCallImpl<ValueT>* composed_call) {
    return InterpretBuiltinMethodCall(
        func, ValueT{adt::Nothing{}}, composed_call);
  }

  Result<adt::Ok> InterpretBuiltinHighOrderFuncCall(
      const BuiltinHighOrderFuncType<ValueT>& func,
      ComposedCallImpl<ValueT>* composed_call) {
    return InterpretBuiltinHighOrderMethodCall(
        func, ValueT{adt::Nothing{}}, composed_call);
  }

  Result<adt::Ok> InterpretBuiltinMethodCall(
      const BuiltinFuncType<ValueT>& func,
      const ValueT& obj,
      ComposedCallImpl<ValueT>* composed_call) {
    const auto original_outter_func = composed_call->outter_func;
    ADT_LET_CONST_REF(inner_ret, func(obj, composed_call->args));
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

  Result<adt::Ok> InterpretBuiltinHighOrderMethodCall(
      const BuiltinHighOrderFuncType<ValueT>& func,
      const ValueT& obj,
      ComposedCallImpl<ValueT>* composed_call) {
    const auto& Apply = [this](const ValueT& func,
                               const std::vector<ValueT>& args) {
      return this->Interpret(func, args);
    };
    const auto original_outter_func = composed_call->outter_func;
    ADT_LET_CONST_REF(inner_ret, func(Apply, obj, composed_call->args));
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
  static Frame<ValueT> GetBuiltinFrame() {
    Object<ValueT> object{ValueT::GetExportedTypes()};
    object->Set("print", &This::Print);
    return Frame<ValueT>{object};
  }

  static adt::Result<ValueT> Print(const ValueT&,
                                   const std::vector<ValueT>& args) {
    std::ostringstream ss;
    int i = 0;
    for (const auto& obj : args) {
      if (i++ > 0) {
        ss << " ";
      }
      const auto& func = MethodClass<ValueT>::ToString(obj);
      ADT_LET_CONST_REF(str_val, func(obj));
      ADT_LET_CONST_REF(str, str_val.template TryGet<std::string>())
          << adt::errors::TypeError{
                 std::string() + "'" + GetTypeName(obj) +
                 ".__builtin_ToString__ should return a 'str' but '" +
                 GetTypeName(str_val) + "' were returned."};
      ss << str;
    }
    LOG(ERROR) << "Print\n" << ss.str();
    return adt::Nothing{};
  }
};

}  // namespace ap::axpr
