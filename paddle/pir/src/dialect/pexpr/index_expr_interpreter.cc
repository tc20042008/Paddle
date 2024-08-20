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
#include "paddle/pir/include/dialect/pexpr/index_expr_interpreter.h"
#include "paddle/pir/include/dialect/pexpr/index_expr_builtin_functions.h"

namespace pexpr::index_expr {

class IndexExprInterpreterImpl {
 public:
  explicit IndexExprInterpreterImpl(EnvMgr* env_mgr)
      : env_mgr_(env_mgr), builtin_frame_(MakeBuiltinFrame()) {}
  IndexExprInterpreterImpl(const IndexExprInterpreterImpl&) = delete;
  IndexExprInterpreterImpl(IndexExprInterpreterImpl&&) = delete;

  Result<Val> operator()(const Closure<Val>& closure,
                         const std::vector<Val>& args) {
    return InterpretClosure(closure, args);
  }

  const Frame<Val>& builtin_frame() const { return builtin_frame_; }

 private:
  Result<Val> Interpret(const CoreExpr& code, const std::shared_ptr<Env>& env) {
    return code.Match(
        [&](const Atomic<CoreExpr>& atomic) {
          return InterpretAtomic(atomic, env);
        },
        [&](const ComposedCall<CoreExpr>& composed_call) {
          return InterpretComposedCall(composed_call, env);
        });
  }

  Result<Val> InterpretAtomic(const Atomic<CoreExpr>& atomic,
                              const std::shared_ptr<Env>& env) {
    return atomic.Match(
        [&](const Lambda<CoreExpr>& lambda) -> Result<Val> {
          return Val{Closure<Val>{lambda, env}};
        },
        [&](const tVar<std::string>& var) -> Result<Val> {
          return env->Get(var.value())
              .Match(
                  [&](const Error& error) -> Result<Val> {
                    return NameError{std::string("name '") + var.value() +
                                     "' is not defined."};
                  },
                  [&](const auto& val) -> Result<Val> { return val; });
        },
        [&](const PrimitiveOp& op) -> Result<Val> {
          return RuntimeError{"primitive op not supported."};
        },
        [&](const auto& val) -> Result<Val> { return Val{val}; });
  }

  Result<Val> InterpretComposedCall(const ComposedCall<CoreExpr>& composed_call,
                                    const std::shared_ptr<Env>& env) {
    Result<Val> inner_func = InterpretAtomic(composed_call->inner_func, env);
    if (inner_func.Has<Error>()) {
      return inner_func.Get<Error>();
    }
    std::vector<Val> arg_values;
    arg_values.reserve(composed_call->args.size());
    for (const auto& arg : composed_call->args) {
      Result<Val> arg_value = InterpretAtomic(arg, env);
      if (arg_value.Has<Error>()) {
        return arg_value.Get<Error>();
      }
      arg_values.push_back(arg_value.Get<Val>());
    }
    Result<Val> inner_ret = InterpretCall(inner_func.Get<Val>(), arg_values);
    if (inner_ret.Has<Error>()) {
      return inner_ret.Get<Error>();
    }
    Result<Val> outter_func = InterpretAtomic(composed_call->outter_func, env);
    if (outter_func.Has<Error>()) {
      return outter_func.Get<Error>();
    }
    return InterpretCall(outter_func.Get<Val>(), {inner_ret.Get<Val>()});
  }

  Result<Val> InterpretClosure(const Closure<Val>& closure,
                               const std::vector<Val>& args) {
    const auto& new_env = env_mgr_->New(closure->environment);
    if (args.size() != closure->lambda->args.size()) {
      return TypeError{std::string("<lambda>() takes ") +
                       std::to_string(closure->lambda->args.size()) +
                       " positional arguments but " +
                       std::to_string(args.size()) + " was given"};
    }
    for (int i = 0; i < args.size(); ++i) {
      const auto& arg_name = closure->lambda->args.at(i).value();
      if (!new_env->Set(arg_name, args.at(i))) {
        return SyntaxError{"duplicate argument '" + arg_name +
                           "' in function definition"};
      }
    }
    return Interpret(closure->lambda->body, new_env);
  }

  Result<Val> InterpretCall(const Val& f, const std::vector<Val>& args) {
    return f.Match(
        [&](const BuiltinFuncType<Val>& func) -> Result<Val> {
          const auto& Func = [this](
                                 const Closure<Val>& closure,
                                 const std::vector<Val>& args) -> Result<Val> {
            return InterpretClosure(closure, args);
          };
          return func(Func, args);
        },
        [&](const Closure<Val>& closure) -> Result<Val> {
          return InterpretClosure(closure, args);
        },
        [&](const auto& other) -> Result<Val> {
          return TypeError{std::string("'") + GetBuiltinTypeName(f) +
                           "' object is not callable"};
        });
  }

  static Frame<Val> MakeBuiltinFrame() { return Frame<Val>{InitBuiltins()}; }

  static Object<Val> InitBuiltins() {
    return Object<Val>{std::unordered_map<std::string, Val>{
        {kIf, Val{BuiltinFuncType<Val>(&BuiltinIf<Val>)}},
        {kBuiltinId, Val{BuiltinFuncType<Val>(&BuiltinIdentity<Val>)}},
        {"__builtin_apply__", Val{BuiltinFuncType<Val>(&BuiltinApply<Val>)}},
        {"list", Val{BuiltinFuncType<Val>(&BuiltinList<Val>)}},
        {"kUndefinedIndexTupleExpr",
         Val{IndexExprValue{IndexTupleExpr{UndefinedIndexTupleExpr{}}}}},
        {"kNothingIndexTupleExpr",
         Val{IndexExprValue{IndexTupleExpr{NothingIndexTupleExpr{}}}}},
        {"kIntArrayLikeIndexTupleExpr",
         Val{IndexExprValue{IndexTupleExpr{IntArrayLikeIndexTupleExpr{}}}}},
        {"kUndefinedIndexExpr",
         Val{IndexExprValue{IndexExpr{UndefinedIndexExpr{}}}}},
        {"PtrGetItem", Val{BuiltinFuncType<Val>(&MakePtrGetItem)}},
        {"IndexExprBroadcastMask",
         Val{BuiltinFuncType<Val>(&MakeIndexExprBroadcastMask)}},
        {"Slice", Val{BuiltinFuncType<Val>(&MakeSlice)}},
        {"IndexExprSlice", Val{BuiltinFuncType<Val>(&MakeIndexExprSlice)}},
        {"IndexExprAffine", Val{BuiltinFuncType<Val>(&MakeIndexExprAffine)}},
        {"DisjointUnion", Val{BuiltinFuncType<Val>(&MakeDisjointUnion)}},
        {"IndexTupleExprPermute",
         Val{BuiltinFuncType<Val>(&MakeIndexTupleExprPermute)}},
        {"IndexTupleExprReshape",
         Val{BuiltinFuncType<Val>(&MakeIndexTupleExprReshape)}},
        {"IndexTupleExprTransform",
         Val{BuiltinFuncType<Val>(&MakeIndexTupleExprTransform)}},
        {"OpIndexTupleExprSignature",
         Val{BuiltinFuncType<Val>(&MakeOpIndexTupleExprSignature)}},
        {"InIndexTupleExprSignature",
         Val{BuiltinFuncType<Val>(&MakeInIndexTupleExprSignature)}},
        {"OutIndexTupleExprSignature",
         Val{BuiltinFuncType<Val>(&MakeOutIndexTupleExprSignature)}},
    }};
  }

  EnvMgr* env_mgr_;
  Frame<Val> builtin_frame_;
};

IndexExprInterpreter::IndexExprInterpreter()
    : env_mgr_(std::make_shared<EnvMgr>()),
      impl_(std::make_unique<IndexExprInterpreterImpl>(env_mgr_.get())) {}

IndexExprInterpreter::IndexExprInterpreter(
    const std::shared_ptr<EnvMgr>& env_mgr)
    : env_mgr_(env_mgr),
      impl_(std::make_unique<IndexExprInterpreterImpl>(env_mgr.get())) {}

Result<Val> IndexExprInterpreter::operator()(const Lambda<CoreExpr>& lambda,
                                             const std::vector<Val>& args) {
  const auto& env = env_mgr_->New(impl_->builtin_frame());
  Closure<Val> closure{lambda, env};
  return (*impl_)(closure, args);
}

Result<Val> IndexExprInterpreter::operator()(
    const std::unordered_map<std::string, BuiltinFuncType<Val>>&
        global_functions,
    const Lambda<CoreExpr>& lambda,
    const std::vector<Val>& args) {
  const auto& env = env_mgr_->New(impl_->builtin_frame());
  for (const auto& [name, val] : global_functions) {
    env->Set(name, Val{val});
  }
  Closure<Val> closure{lambda, env};
  return (*impl_)(closure, args);
}

}  // namespace pexpr::index_expr
