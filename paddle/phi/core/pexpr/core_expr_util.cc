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

#include "paddle/phi/core/pexpr/core_expr_util.h"
#include <atomic>
#include "paddle/common/enforce.h"
#include "paddle/phi/core/pexpr/anf_builder.h"
#include "paddle/phi/core/pexpr/anf_interpreter.h"
#include "paddle/phi/core/pexpr/core_expr_builder.h"
#include "paddle/phi/core/pexpr/value.h"

namespace pexpr {

namespace {

CoreExpr Replace(const CoreExpr& core_expr,
                 const tVar<std::string>& pattern_var,
                 const tVar<std::string>& replacement) {
  return core_expr.Match(
      [&](const Atomic<CoreExpr>& atomic_expr) -> CoreExpr {
        return atomic_expr.Match(
            [&](const Lambda<CoreExpr>& lambda) -> Atomic<CoreExpr> {
              for (const auto& arg : lambda.args) {
                if (arg == pattern_var) {
                  return lambda;
                }
              }
              CoreExpr new_body =
                  Replace(*lambda.body, pattern_var, replacement);
              return CoreExprBuilder().Lambda(lambda.args, new_body);
            },
            [&](const tVar<std::string>& var) -> Atomic<CoreExpr> {
              if (var == pattern_var) {
                return replacement;
              } else {
                return var;
              }
            },
            [&](const auto& expr) -> Atomic<CoreExpr> { return expr; });
      },
      [&](const ComposedCall<CoreExpr>& composed_call) -> CoreExpr {
        const auto& new_outter_func =
            Replace(composed_call.outter_func, pattern_var, replacement);
        if (!new_outter_func.Has<Atomic<CoreExpr>>()) {
          return composed_call;
        }
        const auto& new_inner_func =
            Replace(composed_call.inner_func, pattern_var, replacement);
        if (!new_inner_func.Has<Atomic<CoreExpr>>()) {
          return composed_call;
        }
        std::vector<Atomic<CoreExpr>> new_args;
        new_args.reserve(composed_call.args.size());
        for (const auto& arg : composed_call.args) {
          const auto& new_arg = Replace(arg, pattern_var, replacement);
          if (!new_arg.Has<Atomic<CoreExpr>>()) {
            return composed_call;
          }
          new_args.push_back(new_arg.Get<Atomic<CoreExpr>>());
        }
        return CoreExprBuilder().ComposedCall(
            new_outter_func.Get<Atomic<CoreExpr>>(),
            new_inner_func.Get<Atomic<CoreExpr>>(),
            new_args);
      });
}

std::optional<CoreExpr> TryInlineBuiltinId(
    const ComposedCall<CoreExpr>& composed_call) {
  if (!composed_call.outter_func.Has<Lambda<CoreExpr>>()) {
    return std::nullopt;
  }
  const auto& outter_func = composed_call.outter_func.Get<Lambda<CoreExpr>>();
  if (!composed_call.inner_func.Has<tVar<std::string>>()) {
    return std::nullopt;
  }
  const auto& inner_func_name =
      composed_call.inner_func.Get<tVar<std::string>>().value();
  if (inner_func_name != kBuiltinId) {
    return std::nullopt;
  }
  if (composed_call.args.size() != 1) {
    return std::nullopt;
  }
  if (!composed_call.args.at(0).Has<tVar<std::string>>()) {
    return std::nullopt;
  }
  const auto& pattern_var = outter_func.args.at(0);
  const auto& replacement = composed_call.args.at(0).Get<tVar<std::string>>();
  return Replace(*outter_func.body, pattern_var, replacement);
}

std::optional<int> GetVarArgIndex(const ComposedCall<CoreExpr>& composed_call) {
  for (int i = 0; i < composed_call.args.size(); ++i) {
    if (composed_call.args.at(i).Has<tVar<std::string>>()) {
      return i;
    }
  }
  return std::nullopt;
}

ComposedCall<CoreExpr> TryInlineInnerLambdaArg(
    const ComposedCall<CoreExpr>& composed_call, int arg_idx) {
  const auto& origin_inner_func =
      composed_call.inner_func.Get<Lambda<CoreExpr>>();
  std::vector<tVar<std::string>> inner_func_args;
  inner_func_args.reserve(origin_inner_func.args.size());
  for (int i = 0; i < origin_inner_func.args.size(); ++i) {
    if (i == arg_idx) {
      continue;
    }
    inner_func_args.push_back(origin_inner_func.args.at(i));
  }
  std::vector<Atomic<CoreExpr>> call_args;
  call_args.reserve(composed_call.args.size());
  for (int i = 0; i < composed_call.args.size(); ++i) {
    if (i == arg_idx) {
      continue;
    }
    call_args.push_back(composed_call.args.at(i));
  }
  const auto& inner_func_body =
      Replace(*origin_inner_func.body,
              origin_inner_func.args.at(arg_idx),
              composed_call.args.at(arg_idx).Get<tVar<std::string>>());
  CoreExprBuilder core{};
  const auto& inner_func = core.Lambda(inner_func_args, inner_func_body);
  return core.ComposedCall(composed_call.outter_func, inner_func, call_args);
}

std::optional<CoreExpr> TryInlineInnerLambda(
    const ComposedCall<CoreExpr>& composed_call) {
  if (!composed_call.inner_func.Has<Lambda<CoreExpr>>()) {
    return std::nullopt;
  }
  const auto& inner_func = composed_call.inner_func.Get<Lambda<CoreExpr>>();
  if (inner_func.args.size() != composed_call.args.size()) {
    return std::nullopt;
  }
  if (!GetVarArgIndex(composed_call).has_value()) {
    return std::nullopt;
  }
  ComposedCall<CoreExpr> ret_composed_call = composed_call;
  while (const auto& opt_var_arg_index = GetVarArgIndex(ret_composed_call)) {
    ret_composed_call =
        TryInlineInnerLambdaArg(ret_composed_call, opt_var_arg_index.value());
  }
  return ret_composed_call;
}

Atomic<CoreExpr> InlineAtomic(const Atomic<CoreExpr>& atomic_expr) {
  return atomic_expr.Match(
      [&](const Lambda<CoreExpr>& lambda) -> Atomic<CoreExpr> {
        return CoreExprBuilder().Lambda(lambda.args, Inline(*lambda.body));
      },
      [&](const auto& expr) -> Atomic<CoreExpr> { return expr; });
}

ComposedCall<CoreExpr> InlineComposedCallChildren(
    const ComposedCall<CoreExpr>& composed_call) {
  const auto& outter_func = InlineAtomic(composed_call.outter_func);
  const auto& inner_func = InlineAtomic(composed_call.inner_func);
  std::vector<Atomic<CoreExpr>> args;
  args.reserve(composed_call.args.size());
  for (const auto& arg : composed_call.args) {
    args.push_back(InlineAtomic(arg));
  }
  return CoreExprBuilder().ComposedCall(outter_func, inner_func, args);
}

}  // namespace

CoreExpr Inline(const CoreExpr& core_expr) {
  return core_expr.Match(
      [&](const Atomic<CoreExpr>& atomic_expr) -> CoreExpr {
        return InlineAtomic(atomic_expr);
      },
      [&](const ComposedCall<CoreExpr>& composed_call) -> CoreExpr {
        const auto& new_composed_call =
            InlineComposedCallChildren(composed_call);
        if (const auto& ret = TryInlineBuiltinId(new_composed_call)) {
          return ret.value();
        }
        if (const auto& ret = TryInlineInnerLambda(new_composed_call)) {
          return ret.value();
        }
        return new_composed_call;
      });
}

namespace {

using LazyCoreExpr =
    std::function<ComposedCall<CoreExpr>(const Atomic<CoreExpr>& continuation)>;

using MaybeLazyCoreExprBase = std::variant<CoreExpr, LazyCoreExpr>;

struct MaybeLazyCoreExpr : public MaybeLazyCoreExprBase {
  using MaybeLazyCoreExprBase::MaybeLazyCoreExprBase;

  DEFINE_MATCH_METHOD();

  const MaybeLazyCoreExprBase& variant() const {
    return reinterpret_cast<const MaybeLazyCoreExprBase&>(*this);
  }

  template <typename T>
  bool Has() const {
    return std::holds_alternative<T>(variant());
  }

  template <typename T>
  const T& Get() const {
    return std::get<T>(variant());
  }
};

template <typename T>
Value<AnfExpr, MaybeLazyCoreExpr> CoreVal(const T& val) {
  return Value<AnfExpr, MaybeLazyCoreExpr>{MaybeLazyCoreExpr{CoreExpr{val}}};
}

Value<AnfExpr, MaybeLazyCoreExpr> LazyCoreVal(const LazyCoreExpr& lazy) {
  return Value<AnfExpr, MaybeLazyCoreExpr>{MaybeLazyCoreExpr{lazy}};
}

LazyCoreExpr TryWrapperToLazyCoreExpr(
    const Value<AnfExpr, MaybeLazyCoreExpr>& value);

// Convert anf expr to core expr without duplicate var name.
struct AnfExprToCoreExprConverter {
  AnfExprToCoreExprConverter() : core_() {}

  using custom_value_type = MaybeLazyCoreExpr;

  using interpreter_type = AnfInterpreterBase<custom_value_type>;

  using value_type = Value<AnfExpr, custom_value_type>;

  using env_type = Environment<AnfExpr, custom_value_type>;

  value_type InterpretVar(const tVar<std::string>& anf_expr,
                          const std::shared_ptr<env_type>& env) {
    const auto& val = env->Get(anf_expr.value());
    if (!val.has_value()) {
      return CoreVal(core_.Var(anf_expr.value()));
    }
    PADDLE_ENFORCE_EQ(val.value().Has<MaybeLazyCoreExpr>(),
                      true,
                      phi::errors::InvalidArgument(
                          "Failed to convert anf var to core var. val is not a "
                          "MaybeLazyCoreExpr instance."));
    const auto& may_lazy_core_expr = val.value().Get<MaybeLazyCoreExpr>();
    PADDLE_ENFORCE_EQ(
        may_lazy_core_expr.Has<CoreExpr>(),
        true,
        phi::errors::InvalidArgument("Failed to convert anf var to core var. "
                                     "val is not a CoreExpr instance."));
    const auto& core_expr = may_lazy_core_expr.Get<CoreExpr>();
    PADDLE_ENFORCE_EQ(core_expr.Has<Atomic<CoreExpr>>(),
                      true,
                      phi::errors::InvalidArgument(
                          "Failed to convert anf var to core var. "
                          "val is not a Atomic<CoreExpr> instance."));
    const auto& atomic_expr = core_expr.Get<Atomic<CoreExpr>>();
    PADDLE_ENFORCE_EQ(atomic_expr.Has<tVar<std::string>>(),
                      true,
                      phi::errors::InvalidArgument(
                          "Failed to convert anf var to core var. val is not a "
                          "tVar<std::string> instance."));
    return val.value();
  }

  value_type InterpretBool(const bool c, const std::shared_ptr<env_type>& env) {
    return CoreVal(core_.Bool(c));
  }
  value_type InterpretInt64(const int64_t c,
                            const std::shared_ptr<env_type>& env) {
    return CoreVal(core_.Int64(c));
  }
  value_type InterpretString(const std::string& c,
                             const std::shared_ptr<env_type>& env) {
    return CoreVal(core_.String(c));
  }
  value_type InterpretPrimitiveOp(const PrimitiveOp& c,
                                  const std::shared_ptr<env_type>& env) {
    return CoreVal(core_.PrimitiveOp(c));
  }
  value_type InterpretLambda(const Lambda<AnfExpr>& anf_expr,
                             const std::shared_ptr<env_type>& env,
                             interpreter_type* interpreter) {
    auto new_env = interpreter->NewEnv(env);
    const auto& core_body_val = interpreter->Interpret(*anf_expr.body, new_env);
    LazyCoreExpr lazy_core_expr = TryWrapperToLazyCoreExpr(core_body_val);
    CoreExpr core_body = lazy_core_expr(core_.Var(kBuiltinId));
    return CoreVal(core_.Lambda(anf_expr.args, core_body));
  }

  value_type InterpretCall(const Call<AnfExpr>& anf_expr,
                           const std::shared_ptr<env_type>& env,
                           interpreter_type* interpreter) {
    const auto& inner_func = InterpretAtomic(anf_expr.func, env, interpreter);
    std::vector<Atomic<CoreExpr>> core_args{};
    core_args.reserve(anf_expr.args.size());
    for (const auto& arg : anf_expr.args) {
      core_args.push_back(InterpretAtomic(arg, env, interpreter));
    }
    return LazyCoreVal(
        [inner_func, core_args](const Atomic<CoreExpr>& continuation) {
          CoreExprBuilder core{};
          return core.ComposedCall(continuation, inner_func, core_args);
        });
  }
  value_type InterpretIf(const If<AnfExpr>& anf_expr,
                         const std::shared_ptr<env_type>& env,
                         interpreter_type* interpreter) {
    const Atomic<CoreExpr>& core_cond =
        InterpretAtomic(anf_expr.cond, env, interpreter);
    const auto& MakeZeroArgLambda = [](const auto& expr_ptr) {
      return AnfExprBuilder().Lambda({}, *expr_ptr);
    };
    const Atomic<CoreExpr>& core_true_expr = InterpretAtomic(
        MakeZeroArgLambda(anf_expr.true_expr), env, interpreter);
    const Atomic<CoreExpr>& core_false_expr = InterpretAtomic(
        MakeZeroArgLambda(anf_expr.false_expr), env, interpreter);
    return LazyCoreVal([=](const Atomic<CoreExpr>& continuation) {
      CoreExprBuilder core{};
      return core.ComposedCall(continuation,
                               core.Var("if"),
                               {core_cond, core_true_expr, core_false_expr});
    });
  }
  value_type InterpretLet(const Let<AnfExpr>& anf_expr,
                          const std::shared_ptr<env_type>& env,
                          interpreter_type* interpreter) {
    auto new_env = interpreter->NewEnv(env);
    std::vector<std::string> symbol_names;
    std::vector<LazyCoreExpr> lazy_core_exprs;
    lazy_core_exprs.reserve(anf_expr.bindings.size());
    for (const auto& binding : anf_expr.bindings) {
      symbol_names.push_back(binding.var.value());
      lazy_core_exprs.push_back(
          InterpretCombined(binding.val, new_env, interpreter));
    }
    value_type body_val = interpreter->Interpret(*anf_expr.body, new_env);
    LazyCoreExpr body_lazy_core_expr = TryWrapperToLazyCoreExpr(body_val);
    lazy_core_exprs.push_back(body_lazy_core_expr);
    PADDLE_ENFORCE_EQ(
        lazy_core_exprs.size(),
        symbol_names.size() + 1,
        phi::errors::InvalidArgument(
            "lazy_core_exprs.size() should equal to symbol_names.size() + 1"));
    return LazyCoreVal(
        [symbol_names, lazy_core_exprs](Atomic<CoreExpr> continuation) {
          CoreExprBuilder core{};
          LazyCoreExpr first_body_lazy_core_expr = lazy_core_exprs.at(0);
          for (int i = lazy_core_exprs.size() - 1; i > 0; i--) {
            const auto& var = symbol_names.at(i - 1);
            LazyCoreExpr lazy_core_expr = lazy_core_exprs.at(i);
            CoreExpr body = lazy_core_expr(continuation);
            continuation = core.Lambda({tVar<std::string>{var}}, body);
          }
          return first_body_lazy_core_expr(continuation);
        });
  }

 private:
  void CheckIsAtomic(const value_type& val) {
    PADDLE_ENFORCE_EQ(
        val.Has<MaybeLazyCoreExpr>(),
        true,
        phi::errors::InvalidArgument(
            "InterpretAtomic should return a MaybeLazyCoreExpr instance"));
    const auto& maybe_lazy_core_expr = val.Get<MaybeLazyCoreExpr>();
    PADDLE_ENFORCE_EQ(maybe_lazy_core_expr.Has<CoreExpr>(),
                      true,
                      phi::errors::InvalidArgument(
                          "InterpretAtomic should return a CoreExpr instance"));
    const auto& core_expr = val.Get<MaybeLazyCoreExpr>().Get<CoreExpr>();
    PADDLE_ENFORCE_EQ(
        core_expr.Has<Atomic<CoreExpr>>(),
        true,
        phi::errors::InvalidArgument(
            "InterpretAtomic should return a Atomic<CoreExpr> instance"));
  }

  Atomic<CoreExpr> GetAtomic(const value_type& val) {
    return val.Get<MaybeLazyCoreExpr>().Get<CoreExpr>().Get<Atomic<CoreExpr>>();
  }

  Atomic<CoreExpr> InterpretAtomic(const Atomic<AnfExpr>& atomic_anf,
                                   const std::shared_ptr<env_type>& env,
                                   interpreter_type* interpreter) {
    value_type val = interpreter->InterpretAtomic(atomic_anf, env);
    CheckIsAtomic(val);
    return GetAtomic(val);
  }

  void CheckIsLazyCoreExpr(const value_type& val) {
    PADDLE_ENFORCE_EQ(
        val.Has<MaybeLazyCoreExpr>(),
        true,
        phi::errors::InvalidArgument(
            "InterpretCombined should return a MaybeLazyCoreExpr instance"));
    const auto& maybe_lazy_core_expr = val.Get<MaybeLazyCoreExpr>();
    PADDLE_ENFORCE_EQ(
        maybe_lazy_core_expr.Has<LazyCoreExpr>(),
        true,
        phi::errors::InvalidArgument(
            "InterpretCombined should return a LazyCoreExpr instance"));
  }

  LazyCoreExpr GetLazyCoreExpr(const value_type& val) {
    return val.Get<MaybeLazyCoreExpr>().Get<LazyCoreExpr>();
  }

  LazyCoreExpr InterpretCombined(const Combined<AnfExpr>& combined_anf,
                                 const std::shared_ptr<env_type>& env,
                                 interpreter_type* interpreter) {
    value_type val = interpreter->InterpretCombined(combined_anf, env);
    CheckIsLazyCoreExpr(val);
    return GetLazyCoreExpr(val);
  }

  CoreExprBuilder core_;
};

LazyCoreExpr TryWrapperToLazyCoreExpr(
    const Value<AnfExpr, MaybeLazyCoreExpr>& value) {
  PADDLE_ENFORCE_EQ(
      value.Has<MaybeLazyCoreExpr>(),
      true,
      phi::errors::InvalidArgument(
          "InterpretAtomic should return a MaybeLazyCoreExpr instance"));
  return value.Get<MaybeLazyCoreExpr>().Match(
      [&](const LazyCoreExpr& lazy) { return lazy; },
      [&](const CoreExpr& core_expr) {
        PADDLE_ENFORCE_EQ(
            core_expr.Has<Atomic<CoreExpr>>(),
            true,
            phi::errors::InvalidArgument(
                "core_expr should return a Atomic<CoreExpr> instance"));
        const Atomic<CoreExpr> val = core_expr.Get<Atomic<CoreExpr>>();
        return LazyCoreExpr([val](const Atomic<CoreExpr>& continuation) {
          CoreExprBuilder core{};
          return core.ComposedCall(continuation, core.Var(kBuiltinId), {val});
        });
      });
}

}  // namespace

CoreExpr ConvertAnfExprToCoreExpr(const AnfExpr& anf_expr) {
  AnfExprToCoreExprConverter ctx{};
  EnvironmentManager<AnfExpr, MaybeLazyCoreExpr> env_mgr;
  AnfInterpreter<AnfExprToCoreExprConverter, MaybeLazyCoreExpr> interpreter(
      &ctx, &env_mgr);
  using Val = Value<AnfExpr, MaybeLazyCoreExpr>;
  const auto& argv = Val{
      MaybeLazyCoreExpr{CoreExpr{Atomic<CoreExpr>{tVar<std::string>{"ARGV"}}}}};
  Val ret_val = interpreter(anf_expr, argv);
  const auto& lazy_core_expr = TryWrapperToLazyCoreExpr(ret_val);
  return lazy_core_expr(CoreExprBuilder().Var(kBuiltinId));
}

}  // namespace pexpr
