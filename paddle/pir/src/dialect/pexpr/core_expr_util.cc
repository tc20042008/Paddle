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

#include "paddle/pir/include/dialect/pexpr/core_expr_util.h"
#include <atomic>
#include "paddle/common/enforce.h"
#include "paddle/pir/include/dialect/pexpr/anf_expr_builder.h"
#include "paddle/pir/include/dialect/pexpr/core_expr_builder.h"
#include "paddle/pir/include/dialect/pexpr/value.h"

namespace pexpr {

namespace {

CoreExpr Replace(const CoreExpr& core_expr,
                 const tVar<std::string>& pattern_var,
                 const tVar<std::string>& replacement) {
  return core_expr.Match(
      [&](const Atomic<CoreExpr>& atomic_expr) -> CoreExpr {
        return atomic_expr.Match(
            [&](const Lambda<CoreExpr>& lambda) -> Atomic<CoreExpr> {
              for (const auto& arg : lambda->args) {
                if (arg == pattern_var) {
                  return lambda;
                }
              }
              CoreExpr new_body =
                  Replace(lambda->body, pattern_var, replacement);
              return CoreExprBuilder().Lambda(lambda->args, new_body);
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
      [&](const ComposedCallAtomic<CoreExpr>& composed_call) -> CoreExpr {
        const auto& new_outter_func =
            Replace(composed_call->outter_func, pattern_var, replacement);
        if (!new_outter_func.Has<Atomic<CoreExpr>>()) {
          return composed_call;
        }
        const auto& new_inner_func =
            Replace(composed_call->inner_func, pattern_var, replacement);
        if (!new_inner_func.Has<Atomic<CoreExpr>>()) {
          return composed_call;
        }
        std::vector<Atomic<CoreExpr>> new_args;
        new_args.reserve(composed_call->args.size());
        for (const auto& arg : composed_call->args) {
          const auto& new_arg = Replace(arg, pattern_var, replacement);
          if (!new_arg.Has<Atomic<CoreExpr>>()) {
            return composed_call;
          }
          new_args.push_back(new_arg.Get<Atomic<CoreExpr>>());
        }
        return CoreExprBuilder().ComposedCallAtomic(
            new_outter_func.Get<Atomic<CoreExpr>>(),
            new_inner_func.Get<Atomic<CoreExpr>>(),
            new_args);
      });
}

bool LambdaArgsContains(const Lambda<CoreExpr>& lambda,
                        const std::string& var) {
  for (const auto& arg : lambda->args) {
    if (arg.value() == var) {
      return true;
    }
  }
  return false;
}

Atomic<CoreExpr> ReplaceLambdaArgNameAtomic(
    const Atomic<CoreExpr>& atomic_expr,
    const std::string& pattern_arg_name,
    const std::function<std::string()>& UniqueVarNameGetter) {
  return atomic_expr.Match(
      [&](const Lambda<CoreExpr>& lambda) -> Atomic<CoreExpr> {
        const auto& body = ReplaceLambdaArgName(
            lambda->body, pattern_arg_name, UniqueVarNameGetter);
        CoreExprBuilder core{};
        if (!LambdaArgsContains(lambda, pattern_arg_name)) {
          return core.Lambda(lambda->args, body);
        }
        const auto& replacement_arg_name = UniqueVarNameGetter();
        std::vector<tVar<std::string>> new_args;
        new_args.reserve(lambda->args.size());
        for (const auto& arg : lambda->args) {
          if (arg.value() == pattern_arg_name) {
            new_args.emplace_back(replacement_arg_name);
          } else {
            new_args.emplace_back(arg);
          }
        }
        tVar<std::string> pattern(pattern_arg_name);
        tVar<std::string> replacement(replacement_arg_name);
        return core.Lambda(new_args, Replace(body, pattern, replacement));
      },
      [&](const auto& expr) -> Atomic<CoreExpr> { return expr; });
}

CoreExpr ReplaceLambdaArgNameComposedCallChildren(
    const ComposedCallAtomic<CoreExpr>& core_expr,
    const std::string& pattern_arg_name,
    const std::function<std::string()>& UniqueVarNameGetter) {
  const auto& outter_func = ReplaceLambdaArgNameAtomic(
      core_expr->outter_func, pattern_arg_name, UniqueVarNameGetter);
  const auto& inner_func = ReplaceLambdaArgNameAtomic(
      core_expr->inner_func, pattern_arg_name, UniqueVarNameGetter);
  std::vector<Atomic<CoreExpr>> args;
  args.reserve(core_expr->args.size());
  for (const auto& arg : core_expr->args) {
    args.push_back(
        ReplaceLambdaArgNameAtomic(arg, pattern_arg_name, UniqueVarNameGetter));
  }
  return CoreExprBuilder().ComposedCallAtomic(outter_func, inner_func, args);
}

}  // namespace

CoreExpr ReplaceLambdaArgName(
    const CoreExpr& core_expr,
    const std::string& pattern_arg_name,
    const std::function<std::string()>& UniqueVarNameGetter) {
  return core_expr.Match(
      [&](const Atomic<CoreExpr>& atomic_expr) -> CoreExpr {
        return ReplaceLambdaArgNameAtomic(
            atomic_expr, pattern_arg_name, UniqueVarNameGetter);
      },
      [&](const ComposedCallAtomic<CoreExpr>& composed_call) -> CoreExpr {
        return ReplaceLambdaArgNameComposedCallChildren(
            composed_call, pattern_arg_name, UniqueVarNameGetter);
      });
}
namespace {

std::optional<CoreExpr> TryInlineBuiltinId(
    const ComposedCallAtomic<CoreExpr>& composed_call) {
  if (!composed_call->outter_func.Has<Lambda<CoreExpr>>()) {
    return std::nullopt;
  }
  const auto& outter_func = composed_call->outter_func.Get<Lambda<CoreExpr>>();
  if (!composed_call->inner_func.Has<tVar<std::string>>()) {
    return std::nullopt;
  }
  const auto& inner_func_name =
      composed_call->inner_func.Get<tVar<std::string>>().value();
  if (inner_func_name != kBuiltinId()) {
    return std::nullopt;
  }
  if (composed_call->args.size() != 1) {
    return std::nullopt;
  }
  if (!composed_call->args.at(0).Has<tVar<std::string>>()) {
    return std::nullopt;
  }
  const auto& pattern_var = outter_func->args.at(0);
  const auto& replacement = composed_call->args.at(0).Get<tVar<std::string>>();
  return Replace(outter_func->body, pattern_var, replacement);
}

std::optional<int> GetVarArgIndex(
    const ComposedCallAtomic<CoreExpr>& composed_call) {
  for (int i = 0; i < composed_call->args.size(); ++i) {
    if (composed_call->args.at(i).Has<tVar<std::string>>()) {
      return i;
    }
  }
  return std::nullopt;
}

ComposedCallAtomic<CoreExpr> TryInlineInnerLambdaArg(
    const ComposedCallAtomic<CoreExpr>& composed_call, int arg_idx) {
  const auto& origin_inner_func =
      composed_call->inner_func.Get<Lambda<CoreExpr>>();
  std::vector<tVar<std::string>> inner_func_args;
  inner_func_args.reserve(origin_inner_func->args.size());
  for (int i = 0; i < origin_inner_func->args.size(); ++i) {
    if (i == arg_idx) {
      continue;
    }
    inner_func_args.push_back(origin_inner_func->args.at(i));
  }
  std::vector<Atomic<CoreExpr>> call_args;
  call_args.reserve(composed_call->args.size());
  for (int i = 0; i < composed_call->args.size(); ++i) {
    if (i == arg_idx) {
      continue;
    }
    call_args.push_back(composed_call->args.at(i));
  }
  const auto& inner_func_body =
      Replace(origin_inner_func->body,
              origin_inner_func->args.at(arg_idx),
              composed_call->args.at(arg_idx).Get<tVar<std::string>>());
  CoreExprBuilder core{};
  const auto& inner_func = core.Lambda(inner_func_args, inner_func_body);
  return core.ComposedCallAtomic(
      composed_call->outter_func, inner_func, call_args);
}

std::optional<CoreExpr> TryInlineInnerLambda(
    const ComposedCallAtomic<CoreExpr>& composed_call) {
  if (!composed_call->inner_func.Has<Lambda<CoreExpr>>()) {
    return std::nullopt;
  }
  const auto& inner_func = composed_call->inner_func.Get<Lambda<CoreExpr>>();
  if (inner_func->args.size() != composed_call->args.size()) {
    return std::nullopt;
  }
  if (!GetVarArgIndex(composed_call).has_value()) {
    return std::nullopt;
  }
  ComposedCallAtomic<CoreExpr> ret_composed_call = composed_call;
  while (const auto& opt_var_arg_index = GetVarArgIndex(ret_composed_call)) {
    ret_composed_call =
        TryInlineInnerLambdaArg(ret_composed_call, opt_var_arg_index.value());
  }
  return ret_composed_call;
}

Atomic<CoreExpr> InlineAtomic(const Atomic<CoreExpr>& atomic_expr) {
  return atomic_expr.Match(
      [&](const Lambda<CoreExpr>& lambda) -> Atomic<CoreExpr> {
        return CoreExprBuilder().Lambda(lambda->args, Inline(lambda->body));
      },
      [&](const auto& expr) -> Atomic<CoreExpr> { return expr; });
}

ComposedCallAtomic<CoreExpr> InlineComposedCallChildren(
    const ComposedCallAtomic<CoreExpr>& composed_call) {
  const auto& outter_func = InlineAtomic(composed_call->outter_func);
  const auto& inner_func = InlineAtomic(composed_call->inner_func);
  std::vector<Atomic<CoreExpr>> args;
  args.reserve(composed_call->args.size());
  for (const auto& arg : composed_call->args) {
    args.push_back(InlineAtomic(arg));
  }
  return CoreExprBuilder().ComposedCallAtomic(outter_func, inner_func, args);
}

}  // namespace

CoreExpr Inline(const CoreExpr& core_expr) {
  return core_expr.Match(
      [&](const Atomic<CoreExpr>& atomic_expr) -> CoreExpr {
        return InlineAtomic(atomic_expr);
      },
      [&](const ComposedCallAtomic<CoreExpr>& composed_call) -> CoreExpr {
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

}  // namespace pexpr
