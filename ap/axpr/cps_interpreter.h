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
#include "ap/axpr/builtin_environment.h"
#include "ap/axpr/builtin_functions.h"
#include "ap/axpr/call_environment.h"
#include "ap/axpr/const_global_environment.h"
#include "ap/axpr/core_expr.h"
#include "ap/axpr/error.h"
#include "ap/axpr/mutable_global_environment.h"
#include "ap/axpr/to_string.h"
#include "ap/axpr/value.h"
#include "ap/axpr/value_method_class.h"
#include "ap/memory/thread_local_circlable_ref_list_helper.h"

namespace ap::axpr {

template <typename ValueT>
class CpsInterpreter : public CpsInterpreterBase<ValueT> {
 public:
  using This = CpsInterpreter;
  using Env = Environment<ValueT>;
  CpsInterpreter()
      : builtin_env_(GetBuiltinEnvironment()),
        circlable_ref_list_(std::make_shared<memory::CirclableRefList>()) {}
  CpsInterpreter(const CpsInterpreter&) = delete;
  CpsInterpreter(CpsInterpreter&&) = delete;

  const std::shared_ptr<Env>& builtin_env() const { return builtin_env_; }

  Result<ValueT> Interpret(const Lambda<CoreExpr>& lambda,
                           const std::vector<ValueT>& args) {
    Function<SerializableValue> function{lambda, std::nullopt};
    memory::ThreadLocalCirclableRefListHelper helper{};
    return helper.Guard([&] { return ProtectedInterpret(function, args); });
  }

  Result<ValueT> Interpret(const Function<SerializableValue>& function,
                           const std::vector<ValueT>& args) {
    memory::ThreadLocalCirclableRefListHelper helper{};
    return helper.Guard([&] { return ProtectedInterpret(function, args); });
  }

  Result<ValueT> InterpretModule(
      const Frame<SerializableValue>& const_global_frame,
      const Lambda<CoreExpr>& lambda) {
    memory::ThreadLocalCirclableRefListHelper helper{};
    return helper.Guard(
        [&] { return ProtectedInterpretModule(const_global_frame, lambda); });
  }

 protected:
  Result<ValueT> ProtectedInterpret(const Function<SerializableValue>& function,
                                    const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(closure, ConvertFunctionToClosure(function));
    return ProtectedInterpretCall(closure, args);
  }

  Result<ValueT> ProtectedInterpretModule(
      const Frame<SerializableValue>& const_global_frame,
      const Lambda<CoreExpr>& lambda) {
    std::optional<std::shared_ptr<Environment<ValueT>>> env;
    {
      auto tmp_frame_object = std::make_shared<BuiltinObjectImpl<ValueT>>();
      auto tmp_frame =
          Frame<ValueT>::Make(circlable_ref_list_, tmp_frame_object);
      const auto& mut_global_env = MakeMutableGlobalEnvironment(
          builtin_env(), const_global_frame, tmp_frame);
      env = mut_global_env;
    }
    ADT_CHECK(lambda->args.empty());
    Continuation<ValueT> continuation{lambda, env.value()};
    const auto& ret = ProtectedInterpretCall(continuation, {});
    return ret;
  }

  Result<ValueT> ProtectedInterpretCall(const ValueT& func,
                                        const std::vector<ValueT>& args) {
    ComposedCallImpl<ValueT> composed_call{&BuiltinHalt<ValueT>, func, args};
    ADT_RETURN_IF_ERR(InterpretComposedCallUntilHalt(&composed_call));
    ADT_CHECK(IsHalt(composed_call.inner_func))
        << RuntimeError{"CpsInterpreter does not halt."};
    ADT_CHECK(composed_call.args.size() == 1) << RuntimeError{
        std::string() + "halt function takes 1 argument. but " +
        std::to_string(composed_call.args.size()) + " were given."};
    return composed_call.args.at(0);
  }

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
        [&](const Continuation<ValueT>& continuation) -> Result<adt::Ok> {
          return InterpretContinuation(
              &BuiltinHalt<ValueT>, continuation, composed_call);
        },
        [&](const Function<SerializableValue>& function) -> Result<adt::Ok> {
          ADT_LET_CONST_REF(closure, ConvertFunctionToClosure(function));
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
              std::string("'") + axpr::GetTypeName(composed_call->inner_func) +
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
          if (const auto& const_global_frame = env->GetConstGlobalFrame()) {
            return Function<SerializableValue>{lambda,
                                               const_global_frame.value()};
          } else {
            return Closure<ValueT>{lambda, env};
          }
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

  Result<ValueT> InterpretAtomicAsContinuation(const std::shared_ptr<Env>& env,
                                               const Atomic<CoreExpr>& atomic) {
    return atomic.Match(
        [&](const Lambda<CoreExpr>& lambda) -> Result<ValueT> {
          return Continuation<ValueT>{lambda, env};
        },
        [&](const Symbol& symbol) -> Result<ValueT> {
          return symbol.Match(
              [&](const tVar<std::string>& var) -> Result<ValueT> {
                ADT_CHECK(var.value() == kBuiltinReturn());
                ADT_LET_CONST_REF(val, env->Get(var.value()))
                    << adt::errors::NotImplementedError{
                           "no return continuation found."};
                return val;
              },
              [&](const auto&) -> Result<ValueT> {
                return adt::errors::NotImplementedError{
                    "Invalid continuation."};
              });
        },
        [&](const auto&) -> Result<ValueT> {
          return adt::errors::NotImplementedError{"Invalid continuation."};
        });
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
                     axpr::GetTypeName(operand) + "'"};
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
                     axpr::GetTypeName(lhs) + "'"};
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
    const auto& new_env = MakeCallEnvironment(closure->environment);
    ADT_RETURN_IF_ERR(new_env->Set(kBuiltinReturn(), continuation));
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
      ADT_RETURN_IF_ERR(env->Set(arg_name, args.at(i)));
    }
    return InterpretLambdaBody(
        env, outter_func, lambda->body, ret_composed_call);
  }

  Result<adt::Ok> InterpretContinuation(
      const ValueT& outter_func,
      const Continuation<ValueT>& continuation,
      ComposedCallImpl<ValueT>* composed_call) {
    const auto& env = continuation->environment;
    const auto& lambda = continuation->lambda;
    if (lambda->args.size() > 0) {
      ADT_CHECK(lambda->args.size() == 1);
      ADT_CHECK(composed_call->args.size() == 1);
      ADT_RETURN_IF_ERR(
          env->Set(lambda->args.at(0).value(), composed_call->args.at(0)));
    } else {
      // Do nothing.
    }
    return InterpretLambdaBody(env, outter_func, lambda->body, composed_call);
  }

  Result<adt::Ok> InterpretLambdaBody(
      const std::shared_ptr<Env>& env,
      const ValueT& outter_func,
      const CoreExpr& lambda_body,
      ComposedCallImpl<ValueT>* ret_composed_call) {
    return lambda_body.Match(
        [&](const Atomic<CoreExpr>& atomic) -> Result<adt::Ok> {
          ADT_LET_CONST_REF(val, InterpretAtomic(env, atomic));
          ret_composed_call->inner_func = outter_func;
          ret_composed_call->outter_func = &BuiltinHalt<ValueT>;
          ret_composed_call->args = {val};
          return adt::Ok{};
        },
        [&](const ComposedCallAtomic<CoreExpr>& core_expr) -> Result<adt::Ok> {
          return InterpretLambdaBodyComposedCallAtomic(
              env, core_expr, ret_composed_call);
        });
  }

  Result<adt::Ok> InterpretLambdaBodyComposedCallAtomic(
      const std::shared_ptr<Env>& env,
      const ComposedCallAtomic<CoreExpr>& core_expr,
      ComposedCallImpl<ValueT>* ret_composed_call) {
    ADT_LET_CONST_REF(
        continuation,
        InterpretAtomicAsContinuation(env, core_expr->outter_func));
    ADT_LET_CONST_REF(new_inner_func,
                      InterpretAtomic(env, core_expr->inner_func));
    std::vector<ValueT> args;
    args.reserve(core_expr->args.size());
    for (const auto& arg_expr : core_expr->args) {
      ADT_LET_CONST_REF(arg, InterpretAtomic(env, arg_expr));
      args.emplace_back(arg);
    }
    ret_composed_call->outter_func = continuation;
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
    ADT_LET_CONST_REF(inner_ret, func(obj, composed_call->args));
    composed_call->inner_func = composed_call->outter_func;
    composed_call->outter_func = &BuiltinHalt<ValueT>;
    composed_call->args = {inner_ret};
    return adt::Ok{};
  }

  Result<adt::Ok> InterpretBuiltinHighOrderMethodCall(
      const BuiltinHighOrderFuncType<ValueT>& func,
      const ValueT& obj,
      ComposedCallImpl<ValueT>* composed_call) {
    const auto& Apply = [this](const ValueT& func,
                               const std::vector<ValueT>& args) {
      return this->ProtectedInterpretCall(func, args);
    };
    ADT_LET_CONST_REF(inner_ret, func(Apply, obj, composed_call->args));
    composed_call->inner_func = composed_call->outter_func;
    composed_call->outter_func = &BuiltinHalt<ValueT>;
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

  std::shared_ptr<Env> builtin_env_;
  std::shared_ptr<memory::CirclableRefListBase> circlable_ref_list_;

 private:
  Result<Closure<ValueT>> ConvertFunctionToClosure(
      const Function<SerializableValue>& function) {
    const auto& global_frame = function->global_frame;
    if (global_frame.has_value()) {
      const auto& const_env =
          MakeConstGlobalEnvironment(builtin_env(), global_frame.value());
      return Closure<ValueT>{function->lambda, const_env};
    } else {
      return Closure<ValueT>{function->lambda, builtin_env()};
    }
  }

  static std::shared_ptr<Environment<ValueT>> GetBuiltinEnvironment() {
    return std::make_shared<BuiltinEnvironment<ValueT>>(
        GetBuiltinFrameObject());
  }

  static std::shared_ptr<Environment<ValueT>> MakeConstGlobalEnvironment(
      const std::shared_ptr<Environment<ValueT>>& parent,
      const Frame<SerializableValue>& frame) {
    return std::make_shared<ConstGlobalEnvironment<ValueT>>(parent, frame);
  }

  static std::shared_ptr<Environment<ValueT>> MakeMutableGlobalEnvironment(
      const std::shared_ptr<Environment<ValueT>>& parent,
      const Frame<SerializableValue>& const_frame,
      const Frame<ValueT>& temp_frame) {
    return std::make_shared<MutableGlobalEnvironment<ValueT>>(
        parent, const_frame, temp_frame);
  }

  std::shared_ptr<Environment<ValueT>> MakeCallEnvironment(
      const std::shared_ptr<Environment<ValueT>>& parent) {
    auto builtin_obj = std::make_shared<BuiltinObjectImpl<ValueT>>();
    const auto& frame = Frame<ValueT>::Make(circlable_ref_list_, builtin_obj);
    return std::make_shared<CallEnvironment<ValueT>>(parent, frame);
  }

  static BuiltinObject<ValueT> GetBuiltinFrameObject() {
    return MakeBuiltinFrameObject();
  }

  static BuiltinObject<ValueT> MakeBuiltinFrameObject() {
    BuiltinObject<ValueT> object{ValueT::GetExportedTypes()};
    object->Set("print", &Print<ValueT>);
    object->Set("replace_or_trim_left_comma", &ReplaceOrTrimLeftComma<ValueT>);
    object->Set("range", &MakeRange<ValueT>);
    object->Set("map", &Map<ValueT>);
    object->Set("filter", &Filter<ValueT>);
    object->Set("reduce", &Reduce<ValueT>);
    object->Set("zip", &Zip<ValueT>);
    object->Set("max", &Max<ValueT>);
    object->Set("min", &Min<ValueT>);
    return object;
  }
};

}  // namespace ap::axpr
