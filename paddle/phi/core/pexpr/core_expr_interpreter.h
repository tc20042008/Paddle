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

#include "paddle/phi/core/pexpr/core_expr.h"
#include "paddle/phi/core/pexpr/value.h"

namespace pexpr {

template <typename CustomValueT>
class CoreExprInterpreterBase {
 public:
  using custom_value_type = CustomValueT;

  using value_type = Value<CoreExpr, custom_value_type>;

  using env_type = Environment<CoreExpr, custom_value_type>;

  using env_mgr_type = EnvironmentManager<CoreExpr, custom_value_type>;

  explicit CoreExprInterpreterBase(env_mgr_type* env_mgr) : env_mgr_(env_mgr) {}
  CoreExprInterpreterBase(const CoreExprInterpreterBase&) = delete;
  CoreExprInterpreterBase(CoreExprInterpreterBase&&) = delete;

  std::shared_ptr<env_type> NewEnv(const std::shared_ptr<env_type>& parent) {
    return env_mgr_->New(parent);
  }

  virtual value_type Interpret(const CoreExpr& core_expr,
                               const std::shared_ptr<env_type>& env) = 0;
  virtual value_type InterpretAtomic(const Atomic<CoreExpr>& core_expr,
                                     const std::shared_ptr<env_type>& env) = 0;

 protected:
  env_mgr_type* env_mgr_;
};

template <typename ContextT, typename CustomValueT>
class CoreExprInterpreter
    : public CoreExprInterpreterBase<typename ContextT::custom_value_type> {
 public:
  using custom_value_type = CustomValueT;

  using value_type = Value<CoreExpr, custom_value_type>;

  using env_type = Environment<CoreExpr, custom_value_type>;

  using env_mgr_type = EnvironmentManager<CoreExpr, custom_value_type>;

  explicit CoreExprInterpreter(ContextT* ctx, env_mgr_type* env_mgr)
      : ctx_(ctx),
        CoreExprInterpreterBase<typename ContextT::custom_value_type>(env_mgr) {
  }
  CoreExprInterpreter(const CoreExprInterpreter&) = delete;
  CoreExprInterpreter(CoreExprInterpreter&&) = delete;

  value_type operator()(const CoreExpr& core_expr, const value_type& argv) {
    const auto& env = this->NewEnv(nullptr);
    env->Set("ARGV", argv);
    return Interpret(core_expr, env);
  }

 private:
  value_type Interpret(const CoreExpr& core_expr,
                       const std::shared_ptr<env_type>& env) override {
    return core_expr.Match(
        [&](const Atomic<CoreExpr>& atomic_expr) {
          return InterpretAtomic(atomic_expr, env);
        },
        [&](const ComposedCall<CoreExpr>& combined_expr) {
          return ctx_->InterpretComposedCall(var, env);
        });
  }

  value_type InterpretAtomic(const Atomic<CoreExpr>& atomic_expr,
                             const std::shared_ptr<env_type>& env) override {
    return atomic_expr.Match(
        [&](const tVar<std::string>& var) {
          return ctx_->InterpretVar(var, env);
        },
        [&](bool c) { return ctx_->InterpretBool(c, env); },
        [&](int64_t c) { return ctx_->InterpretInt64(c, env); },
        [&](const std::string& c) { return ctx_->InterpretString(c, env); },
        [&](const PrimitiveOp& c) {
          return ctx_->InterpretPrimitiveOp(c, env);
        },
        [&](const Lambda<CoreExpr>& lambda) {
          return ctx_->InterpretLambda(lambda, env, self());
        });
  }

  CoreExprInterpreterBase<typename ContextT::custom_value_type>* self() {
    return this;
  }

  ContextT* ctx_;
};

}  // namespace pexpr
