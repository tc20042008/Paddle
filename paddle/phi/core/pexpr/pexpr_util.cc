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

#include "paddle/phi/core/pexpr/pexpr_util.h"
#include <atomic>
#include "paddle/common/enforce.h"
#include "paddle/phi/core/pexpr/anf_builder.h"
#include "paddle/phi/core/pexpr/anf_interpreter.h"
#include "paddle/phi/core/pexpr/cps_builder.h"
#include "paddle/phi/core/pexpr/value.h"

namespace pexpr {

AnfExpr ConvertArrayAttributeToAnfExpr(const pir::ArrayAttribute& attr) {
  LOG(FATAL) << "ConvertArrayAttributeToAnfExpr not implemented.";
}

pir::ArrayAttribute ConvertArrayAttributeToAnfExpr(const AnfExpr&,
                                                   pir::IrContext*) {
  LOG(FATAL) << "ConvertArrayAttributeToAnfExpr not implemented.";
}

namespace {

using LazyCpsExpr =
    std::function<Call<CpsExpr>(const Atomic<CpsExpr>& continuation)>;

using MaybeLazyCpsExprBase = std::variant<CpsExpr, LazyCpsExpr>;

struct MaybeLazyCpsExpr : public MaybeLazyCpsExprBase {
  using MaybeLazyCpsExprBase::MaybeLazyCpsExprBase;

  DEFINE_MATCH_METHOD();

  const MaybeLazyCpsExprBase& variant() const {
    return reinterpret_cast<const MaybeLazyCpsExprBase&>(*this);
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
Value<AnfExpr, MaybeLazyCpsExpr> CpsVal(const T& val) {
  return Value<AnfExpr, MaybeLazyCpsExpr>{MaybeLazyCpsExpr{CpsExpr{val}}};
}

Value<AnfExpr, MaybeLazyCpsExpr> LazyCpsVal(const LazyCpsExpr& lazy) {
  return Value<AnfExpr, MaybeLazyCpsExpr>{MaybeLazyCpsExpr{lazy}};
}

LazyCpsExpr TryWrapperToLazyCpsExpr(
    const Value<AnfExpr, MaybeLazyCpsExpr>& value);

class NameConverter {
 public:
  explicit NameConverter(const std::string& prefix)
      : prefix_(prefix), seq_no_(0) {}

  std::string GenUniqueVarName() {
    return std::string("v") + std::to_string(seq_no_++);
  }

 private:
  std::string prefix_;
  std::atomic<size_t> seq_no_;
};

// Convert anf expr to cps expr without duplicate var name.
struct AnfExprToCpsExprConverter {
  AnfExprToCpsExprConverter()
      : continuation_name_gen_("k"),
        lambda_continuation_arg_name_("__lambda_continuation_arg_name_2024__"),
        cps_() {}

  using custom_value_type = MaybeLazyCpsExpr;

  using interpreter_type = AnfInterpreterBase<custom_value_type>;

  using value_type = Value<AnfExpr, custom_value_type>;

  using env_type = Environment<AnfExpr, custom_value_type>;

  value_type InterpretVar(const tVar<std::string>& anf_expr,
                          const std::shared_ptr<env_type>& env) {
    const auto& val = env->Get(anf_expr.value());
    if (!val.has_value()) {
      return CpsVal(cps_.Var(anf_expr.value()));
    }
    PADDLE_ENFORCE_EQ(val.value().Has<MaybeLazyCpsExpr>(),
                      true,
                      phi::errors::InvalidArgument(
                          "Failed to convert anf var to cps var. val is not a "
                          "MaybeLazyCpsExpr instance."));
    const auto& may_lazy_cps_expr = val.value().Get<MaybeLazyCpsExpr>();
    PADDLE_ENFORCE_EQ(
        may_lazy_cps_expr.Has<CpsExpr>(),
        true,
        phi::errors::InvalidArgument("Failed to convert anf var to cps var. "
                                     "val is not a CpsExpr instance."));
    const auto& cps_expr = may_lazy_cps_expr.Get<CpsExpr>();
    PADDLE_ENFORCE_EQ(
        cps_expr.Has<Atomic<CpsExpr>>(),
        true,
        phi::errors::InvalidArgument("Failed to convert anf var to cps var. "
                                     "val is not a Atomic<CpsExpr> instance."));
    const auto& atomic_expr = cps_expr.Get<Atomic<CpsExpr>>();
    PADDLE_ENFORCE_EQ(atomic_expr.Has<tVar<std::string>>(),
                      true,
                      phi::errors::InvalidArgument(
                          "Failed to convert anf var to cps var. val is not a "
                          "tVar<std::string> instance."));
    return val.value();
  }

  value_type InterpretBool(const bool c, const std::shared_ptr<env_type>& env) {
    return CpsVal(cps_.Bool(c));
  }
  value_type InterpretInt64(const int64_t c,
                            const std::shared_ptr<env_type>& env) {
    return CpsVal(cps_.Int64(c));
  }
  value_type InterpretString(const std::string& c,
                             const std::shared_ptr<env_type>& env) {
    return CpsVal(cps_.String(c));
  }
  value_type InterpretPrimitiveOp(const PrimitiveOp& c,
                                  const std::shared_ptr<env_type>& env) {
    return CpsVal(cps_.PrimitiveOp(c));
  }
  value_type InterpretLambda(const Lambda<AnfExpr>& anf_expr,
                             const std::shared_ptr<env_type>& env,
                             interpreter_type* interpreter) {
    auto new_env = interpreter->NewEnv(env);
    std::vector<tVar<std::string>> new_arg_names{anf_expr.args.begin(),
                                                 anf_expr.args.end()};
    // source anf lambda expr : (lambda [a b c] expr)
    // target cps lambda expr : (lambda [a b c continuation] expr)
    std::string cont_arg_name = continuation_name_gen_.GenUniqueVarName();
    new_arg_names.emplace_back(cont_arg_name);
    const auto& cps_body_val = interpreter->Interpret(*anf_expr.body, new_env);
    LazyCpsExpr lazy_cps_expr = TryWrapperToLazyCpsExpr(cps_body_val);
    CpsExpr cps_body = lazy_cps_expr(cps_.Var(cont_arg_name));
    return CpsVal(cps_.Lambda(new_arg_names, cps_body));
  }

  value_type InterpretCall(const Call<AnfExpr>& anf_expr,
                           const std::shared_ptr<env_type>& env,
                           interpreter_type* interpreter) {
    const auto& cps_func = InterpretAtomic(anf_expr.func, env, interpreter);
    std::vector<Atomic<CpsExpr>> cps_args{};
    cps_args.reserve(anf_expr.args.size());
    for (const auto& arg : anf_expr.args) {
      cps_args.push_back(InterpretAtomic(arg, env, interpreter));
    }
    return LazyCpsVal(
        [cps_func, cps_args](const Atomic<CpsExpr>& continuation) {
          std::vector<Atomic<CpsExpr>> args{cps_args.begin(), cps_args.end()};
          args.push_back(continuation);
          return CpsExprBuilder().Call(cps_func, args);
        });
  }
  value_type InterpretIf(const If<AnfExpr>& anf_expr,
                         const std::shared_ptr<env_type>& env,
                         interpreter_type* interpreter) {
    const Atomic<CpsExpr>& cps_cond =
        InterpretAtomic(anf_expr.cond, env, interpreter);
    const auto& MakeZeroArgLambda = [](const auto& expr_ptr) {
      return AnfExprBuilder().Lambda({}, *expr_ptr);
    };
    const Atomic<CpsExpr>& cps_true_expr = InterpretAtomic(
        MakeZeroArgLambda(anf_expr.true_expr), env, interpreter);
    const Atomic<CpsExpr>& cps_false_expr = InterpretAtomic(
        MakeZeroArgLambda(anf_expr.false_expr), env, interpreter);
    return LazyCpsVal([=](const Atomic<CpsExpr>& continuation) {
      CpsExprBuilder cps{};
      return cps.Call(cps.Var("if"),
                      {cps_cond, cps_true_expr, cps_false_expr, continuation});
    });
  }
  value_type InterpretLet(const Let<AnfExpr>& anf_expr,
                          const std::shared_ptr<env_type>& env,
                          interpreter_type* interpreter) {
    auto new_env = interpreter->NewEnv(env);
    std::vector<std::string> symbol_names;
    std::vector<LazyCpsExpr> lazy_cps_exprs;
    lazy_cps_exprs.reserve(anf_expr.bindings.size());
    for (const auto& binding : anf_expr.bindings) {
      symbol_names.push_back(binding.var.value());
      lazy_cps_exprs.push_back(
          InterpretCombined(binding.val, new_env, interpreter));
    }
    value_type body_val = interpreter->Interpret(*anf_expr.body, new_env);
    LazyCpsExpr body_lazy_cps_expr = TryWrapperToLazyCpsExpr(body_val);
    lazy_cps_exprs.push_back(body_lazy_cps_expr);
    PADDLE_ENFORCE_EQ(
        lazy_cps_exprs.size(),
        symbol_names.size() + 1,
        phi::errors::InvalidArgument(
            "lazy_cps_exprs.size() should equal to symbol_names.size() + 1"));
    return LazyCpsVal(
        [symbol_names, lazy_cps_exprs](Atomic<CpsExpr> continuation) {
          CpsExprBuilder cps{};
          LazyCpsExpr first_body_lazy_cps_expr = lazy_cps_exprs.at(0);
          for (int i = lazy_cps_exprs.size() - 1; i > 0; i--) {
            const auto& var = symbol_names.at(i - 1);
            LazyCpsExpr lazy_cps_expr = lazy_cps_exprs.at(i);
            CpsExpr body = lazy_cps_expr(continuation);
            continuation = cps.Lambda({tVar<std::string>{var}}, body);
          }
          return first_body_lazy_cps_expr(continuation);
        });
  }

 private:
  void CheckIsAtomic(const value_type& val) {
    PADDLE_ENFORCE_EQ(
        val.Has<MaybeLazyCpsExpr>(),
        true,
        phi::errors::InvalidArgument(
            "InterpretAtomic should return a MaybeLazyCpsExpr instance"));
    const auto& maybe_lazy_cps_expr = val.Get<MaybeLazyCpsExpr>();
    PADDLE_ENFORCE_EQ(maybe_lazy_cps_expr.Has<CpsExpr>(),
                      true,
                      phi::errors::InvalidArgument(
                          "InterpretAtomic should return a CpsExpr instance"));
    const auto& cps_expr = val.Get<MaybeLazyCpsExpr>().Get<CpsExpr>();
    PADDLE_ENFORCE_EQ(
        cps_expr.Has<Atomic<CpsExpr>>(),
        true,
        phi::errors::InvalidArgument(
            "InterpretAtomic should return a Atomic<CpsExpr> instance"));
  }

  Atomic<CpsExpr> GetAtomic(const value_type& val) {
    return val.Get<MaybeLazyCpsExpr>().Get<CpsExpr>().Get<Atomic<CpsExpr>>();
  }

  Atomic<CpsExpr> InterpretAtomic(const Atomic<AnfExpr>& atomic_anf,
                                  const std::shared_ptr<env_type>& env,
                                  interpreter_type* interpreter) {
    value_type val = interpreter->InterpretAtomic(atomic_anf, env);
    CheckIsAtomic(val);
    return GetAtomic(val);
  }

  void CheckIsLazyCpsExpr(const value_type& val) {
    PADDLE_ENFORCE_EQ(
        val.Has<MaybeLazyCpsExpr>(),
        true,
        phi::errors::InvalidArgument(
            "InterpretCombined should return a MaybeLazyCpsExpr instance"));
    const auto& maybe_lazy_cps_expr = val.Get<MaybeLazyCpsExpr>();
    PADDLE_ENFORCE_EQ(
        maybe_lazy_cps_expr.Has<LazyCpsExpr>(),
        true,
        phi::errors::InvalidArgument(
            "InterpretCombined should return a LazyCpsExpr instance"));
  }

  LazyCpsExpr GetLazyCpsExpr(const value_type& val) {
    return val.Get<MaybeLazyCpsExpr>().Get<LazyCpsExpr>();
  }

  LazyCpsExpr InterpretCombined(const Combined<AnfExpr>& combined_anf,
                                const std::shared_ptr<env_type>& env,
                                interpreter_type* interpreter) {
    value_type val = interpreter->InterpretCombined(combined_anf, env);
    CheckIsLazyCpsExpr(val);
    return GetLazyCpsExpr(val);
  }

  NameConverter continuation_name_gen_;
  std::string lambda_continuation_arg_name_;
  CpsExprBuilder cps_;
};

LazyCpsExpr TryWrapperToLazyCpsExpr(
    const Value<AnfExpr, MaybeLazyCpsExpr>& value) {
  PADDLE_ENFORCE_EQ(
      value.Has<MaybeLazyCpsExpr>(),
      true,
      phi::errors::InvalidArgument(
          "InterpretAtomic should return a MaybeLazyCpsExpr instance"));
  return value.Get<MaybeLazyCpsExpr>().Match(
      [&](const LazyCpsExpr& lazy) { return lazy; },
      [&](const CpsExpr& cps_expr) {
        PADDLE_ENFORCE_EQ(
            cps_expr.Has<Atomic<CpsExpr>>(),
            true,
            phi::errors::InvalidArgument(
                "cps_expr should return a Atomic<CpsExpr> instance"));
        const Atomic<CpsExpr> val = cps_expr.Get<Atomic<CpsExpr>>();
        return LazyCpsExpr([val](const Atomic<CpsExpr>& continuation) {
          CpsExprBuilder cps{};
          return cps.Call(cps.Var("__builtin_identity__"), {val, continuation});
        });
      });
}

}  // namespace

CpsExpr ConvertAnfExprToCpsExpr(const AnfExpr& anf_expr) {
  AnfExprToCpsExprConverter ctx{};
  EnvironmentManager<AnfExpr, MaybeLazyCpsExpr> env_mgr;
  AnfInterpreter<AnfExprToCpsExprConverter, MaybeLazyCpsExpr> interpreter(
      &ctx, &env_mgr);
  using Val = Value<AnfExpr, MaybeLazyCpsExpr>;
  const auto& argv = Val{
      MaybeLazyCpsExpr{CpsExpr{Atomic<CpsExpr>{tVar<std::string>{"ARGV"}}}}};
  Val ret_val = interpreter(anf_expr, argv);
  const auto& lazy_cps_expr = TryWrapperToLazyCpsExpr(ret_val);
  CpsExprBuilder cps{};
  return cps.Lambda({tVar<std::string>{"return"}},
                    lazy_cps_expr(cps.Var("return")));
}

}  // namespace pexpr
