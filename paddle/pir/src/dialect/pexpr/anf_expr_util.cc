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

#include "paddle/pir/include/dialect/pexpr/anf_expr_util.h"
#include <atomic>
#include "paddle/common/enforce.h"
#include "paddle/pir/include/dialect/pexpr/anf_expr_builder.h"
#include "paddle/pir/include/dialect/pexpr/core_expr_builder.h"
#include "paddle/pir/include/dialect/pexpr/core_expr_util.h"

namespace pexpr {

namespace {

using LazyCoreExpr = std::function<ComposedCallAtomic<CoreExpr>(
    const Atomic<CoreExpr>& continuation)>;

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
MaybeLazyCoreExpr CoreVal(const T& val) {
  return MaybeLazyCoreExpr{CoreExpr{val}};
}

MaybeLazyCoreExpr LazyCoreVal(const LazyCoreExpr& lazy) {
  return MaybeLazyCoreExpr{lazy};
}

LazyCoreExpr TryWrapperToLazyCoreExpr(const MaybeLazyCoreExpr& value);

// Convert anf expr to core expr without duplicate var name.
struct AnfExprToCoreExprConverter {
  AnfExprToCoreExprConverter() : core_() {}

  using value_type = MaybeLazyCoreExpr;

  value_type Convert(const AnfExpr& anf_expr) {
    return anf_expr.Match(
        [&](const Atomic<AnfExpr>& atomic_expr) {
          return ConvertAtomic(atomic_expr);
        },
        [&](const Combined<AnfExpr>& combined_expr) {
          return ConvertCombined(combined_expr);
        },
        [&](const Let<AnfExpr>& let_expr) { return ConvertLet(let_expr); });
  }

  value_type ConvertAtomic(const Atomic<AnfExpr>& atomic_expr) {
    return atomic_expr.Match(
        [&](const tVar<std::string>& var) { return ConvertVar(var); },
        [&](bool c) { return ConvertBool(c); },
        [&](int64_t c) { return ConvertInt64(c); },
        [&](const std::string& c) { return ConvertString(c); },
        [&](const PrimitiveOp& c) { return ConvertPrimitiveOp(c); },
        [&](const Lambda<AnfExpr>& lambda) { return ConvertLambda(lambda); });
  }

  value_type ConvertCombined(const Combined<AnfExpr>& combined_expr) {
    return combined_expr.Match(
        [&](const Call<AnfExpr>& call_expr) { return ConvertCall(call_expr); },
        [&](const If<AnfExpr>& if_expr) { return ConvertIf(if_expr); });
  }

  value_type ConvertVar(const tVar<std::string>& anf_expr) {
    return CoreVal(core_.Var(anf_expr.value()));
  }

  value_type ConvertBool(const bool c) { return CoreVal(core_.Bool(c)); }
  value_type ConvertInt64(const int64_t c) { return CoreVal(core_.Int64(c)); }
  value_type ConvertString(const std::string& c) {
    return CoreVal(core_.String(c));
  }
  value_type ConvertPrimitiveOp(const PrimitiveOp& c) {
    return CoreVal(core_.PrimitiveOp(c));
  }
  value_type ConvertLambda(const Lambda<AnfExpr>& anf_expr) {
    const auto& core_body_val = Convert(anf_expr->body);
    LazyCoreExpr lazy_core_expr = TryWrapperToLazyCoreExpr(core_body_val);
    CoreExpr core_body = lazy_core_expr(core_.Var(kBuiltinId));
    return CoreVal(core_.Lambda(anf_expr->args, core_body));
  }

  value_type ConvertCall(const Call<AnfExpr>& anf_expr) {
    const auto& inner_func = ConvertAtomicToAtomic(anf_expr->func);
    std::vector<Atomic<CoreExpr>> core_args{};
    core_args.reserve(anf_expr->args.size());
    for (const auto& arg : anf_expr->args) {
      core_args.push_back(ConvertAtomicToAtomic(arg));
    }
    return LazyCoreVal(
        [inner_func, core_args](const Atomic<CoreExpr>& continuation) {
          CoreExprBuilder core{};
          return core.ComposedCallAtomic(continuation, inner_func, core_args);
        });
  }
  value_type ConvertIf(const If<AnfExpr>& anf_expr) {
    const Atomic<CoreExpr>& core_cond = ConvertAtomicToAtomic(anf_expr->cond);
    const auto& MakeZeroArgLambda = [](const auto& expr_ptr) {
      return AnfExprBuilder().Lambda({}, expr_ptr);
    };
    const Atomic<CoreExpr>& core_true_expr =
        ConvertAtomicToAtomic(MakeZeroArgLambda(anf_expr->true_expr));
    const Atomic<CoreExpr>& core_false_expr =
        ConvertAtomicToAtomic(MakeZeroArgLambda(anf_expr->false_expr));
    return LazyCoreVal([=](const Atomic<CoreExpr>& continuation) {
      CoreExprBuilder core{};
      return core.ComposedCallAtomic(
          continuation,
          core.Var("if"),
          {core_cond, core_true_expr, core_false_expr});
    });
  }
  value_type ConvertLet(const Let<AnfExpr>& anf_expr) {
    std::vector<std::string> symbol_names;
    std::vector<LazyCoreExpr> lazy_core_exprs;
    lazy_core_exprs.reserve(anf_expr->bindings.size());
    for (const auto& binding : anf_expr->bindings) {
      symbol_names.push_back(binding.var.value());
      lazy_core_exprs.push_back(ConvertCombinedToLazyCoreExpr(binding.val));
    }
    value_type body_val = Convert(anf_expr->body);
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
  void CheckIsAtomic(const value_type& maybe_lazy_core_expr) {
    PADDLE_ENFORCE_EQ(maybe_lazy_core_expr.Has<CoreExpr>(),
                      true,
                      phi::errors::InvalidArgument(
                          "ConvertAtomic should return a CoreExpr instance"));
    const auto& core_expr = maybe_lazy_core_expr.Get<CoreExpr>();
    PADDLE_ENFORCE_EQ(
        core_expr.Has<Atomic<CoreExpr>>(),
        true,
        phi::errors::InvalidArgument(
            "ConvertAtomic should return a Atomic<CoreExpr> instance"));
  }

  Atomic<CoreExpr> GetAtomic(const value_type& val) {
    return val.Get<CoreExpr>().Get<Atomic<CoreExpr>>();
  }

  Atomic<CoreExpr> ConvertAtomicToAtomic(const Atomic<AnfExpr>& atomic_anf) {
    value_type val = ConvertAtomic(atomic_anf);
    CheckIsAtomic(val);
    return GetAtomic(val);
  }

  void CheckIsLazyCoreExpr(const value_type& maybe_lazy_core_expr) {
    PADDLE_ENFORCE_EQ(
        maybe_lazy_core_expr.Has<LazyCoreExpr>(),
        true,
        phi::errors::InvalidArgument(
            "ConvertCombined should return a LazyCoreExpr instance"));
  }

  LazyCoreExpr GetLazyCoreExpr(const value_type& val) {
    return val.Get<LazyCoreExpr>();
  }

  LazyCoreExpr ConvertCombinedToLazyCoreExpr(
      const Combined<AnfExpr>& combined_anf) {
    value_type val = ConvertCombined(combined_anf);
    CheckIsLazyCoreExpr(val);
    return GetLazyCoreExpr(val);
  }

  CoreExprBuilder core_;
};

LazyCoreExpr TryWrapperToLazyCoreExpr(
    const MaybeLazyCoreExpr& maybe_lazy_core_expr) {
  return maybe_lazy_core_expr.Match(
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
          return core.ComposedCallAtomic(
              continuation, core.Var(kBuiltinId), {val});
        });
      });
}

}  // namespace

CoreExpr ConvertAnfExprToCoreExpr(const AnfExpr& anf_expr) {
  AnfExprToCoreExprConverter converter{};
  MaybeLazyCoreExpr ret_val = converter.Convert(anf_expr);
  const auto& lazy_core_expr = TryWrapperToLazyCoreExpr(ret_val);
  CoreExpr ret = lazy_core_expr(CoreExprBuilder().Var(kBuiltinId));
  return ret.Match(
      [&](const Atomic<CoreExpr>&) -> CoreExpr { return ret; },
      [&](const ComposedCallAtomic<CoreExpr>& composed_call) -> CoreExpr {
        Atomic<CoreExpr> identity{tVar<std::string>{kBuiltinId}};
        if (composed_call->outter_func != identity) {
          return composed_call;
        }
        if (composed_call->inner_func != identity) {
          return composed_call;
        }
        if (composed_call->args.size() != 1) {
          return composed_call;
        }
        return composed_call->args.at(0);
      });
}

}  // namespace pexpr
