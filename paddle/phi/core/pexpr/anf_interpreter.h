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

#include "paddle/phi/core/pexpr/anf.h"
#include "paddle/phi/core/pexpr/value.h"

namespace pexpr {

template <typename CustomValueT>
class AnfInterpreterBase {
 public:
  using custom_value_type = CustomValueT;

  using value_type = Value<AnfExpr, custom_value_type>;

  using env_type = Environment<AnfExpr, custom_value_type>;

  using env_mgr_type = EnvironmentManager<AnfExpr, custom_value_type>;

  explicit AnfInterpreterBase(env_mgr_type* env_mgr) : env_mgr_(env_mgr) {}
  AnfInterpreterBase(const AnfInterpreterBase&) = delete;
  AnfInterpreterBase(AnfInterpreterBase&&) = delete;

  std::shared_ptr<env_type> NewEnv(const std::shared_ptr<env_type>& parent) {
    return env_mgr_->New(parent);
  }

  virtual value_type Interpret(const AnfExpr& anf_expr,
                               const std::shared_ptr<env_type>& env) = 0;
  virtual value_type InterpretAtomic(const Atomic<AnfExpr>& anf_expr,
                                     const std::shared_ptr<env_type>& env) = 0;
  virtual value_type InterpretCombined(
      const Combined<AnfExpr>& anf_expr,
      const std::shared_ptr<env_type>& env) = 0;

 protected:
  env_mgr_type* env_mgr_;
};

template <typename ContextT, typename CustomValueT>
class AnfInterpreter
    : public AnfInterpreterBase<typename ContextT::custom_value_type> {
 public:
  using custom_value_type = CustomValueT;

  using value_type = Value<AnfExpr, custom_value_type>;

  using env_type = Environment<AnfExpr, custom_value_type>;

  using env_mgr_type = EnvironmentManager<AnfExpr, custom_value_type>;

  explicit AnfInterpreter(ContextT* ctx, env_mgr_type* env_mgr)
      : ctx_(ctx),
        AnfInterpreterBase<typename ContextT::custom_value_type>(env_mgr) {}
  AnfInterpreter(const AnfInterpreter&) = delete;
  AnfInterpreter(AnfInterpreter&&) = delete;

  value_type operator()(const AnfExpr& anf_expr, const value_type& argv) {
    const auto& env = this->NewEnv(nullptr);
    env->Set("ARGV", argv);
    return Interpret(anf_expr, env);
  }

 private:
  value_type Interpret(const AnfExpr& anf_expr,
                       const std::shared_ptr<env_type>& env) override {
    return anf_expr.Match(
        [&](const Atomic<AnfExpr>& atomic_expr) {
          return InterpretAtomic(atomic_expr, env);
        },
        [&](const Combined<AnfExpr>& combined_expr) {
          return InterpretCombined(combined_expr, env);
        },
        [&](const Let<AnfExpr>& let_expr) {
          return ctx_->InterpretLet(let_expr, env, self());
        });
  }

  value_type InterpretAtomic(const Atomic<AnfExpr>& atomic_expr,
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
        [&](const Lambda<AnfExpr>& lambda) {
          return ctx_->InterpretLambda(lambda, env, self());
        });
  }

  value_type InterpretCombined(const Combined<AnfExpr>& combined_expr,
                               const std::shared_ptr<env_type>& env) override {
    return combined_expr.Match(
        [&](const Call<AnfExpr>& call_expr) {
          return ctx_->InterpretCall(call_expr, env, self());
        },
        [&](const If<AnfExpr>& if_expr) {
          return ctx_->InterpretIf(if_expr, env, self());
        });
  }

  AnfInterpreterBase<typename ContextT::custom_value_type>* self() {
    return this;
  }

  ContextT* ctx_;
};

}  // namespace pexpr
