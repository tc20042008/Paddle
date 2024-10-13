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
#include <atomic>
#include "ap/axpr/anf_expr_builder.h"
#include "ap/axpr/core_expr.h"
#include "paddle/common/enforce.h"

namespace ap::axpr {

class LetContext;

class LetVar {
 public:
  LetVar(const LetVar&) = default;
  LetVar(LetVar&&) = default;

  LetVar& operator=(const LetVar& let_var);
  LetVar& operator=(const AnfExpr& anf_val);

  const std::string& name() const { return name_; }

  explicit operator AnfExpr() const { return tVar<std::string>{name()}; }

  LetVar& Attr(const std::string& atttr_name);
  LetVar& At(int64_t idx);

  template <typename... Args>
  AnfExpr Call(Args&&... args);

 private:
  friend class LetContext;
  LetVar(LetContext* let_ctx, const std::string& name)
      : let_ctx_(let_ctx), name_(name) {}

  LetContext* let_ctx_;
  std::string name_;
};

class LetContext : public AtomicExprBuilder<AnfExpr> {
 public:
  explicit LetContext(const std::function<size_t()>& SeqNoGenerator)
      : SeqNoGenerator_(SeqNoGenerator) {}
  LetContext(const LetContext&) = delete;
  LetContext(LetContext&&) = delete;

  using var_type = LetVar;

  LetVar& Var(const std::string& name) {
    auto iter = let_var_storage_.find(name);
    if (iter == let_var_storage_.end()) {
      auto var = std::unique_ptr<LetVar>(new LetVar(this, name));
      iter = let_var_storage_.emplace(name, std::move(var)).first;
    }
    return *iter->second;
  }

  using CallArgBase = std::variant<LetVar, AnfExpr>;

  struct CallArg : public CallArgBase {
    using CallArgBase::CallArgBase;
    const CallArgBase& variant() const {
      return reinterpret_cast<const CallArgBase&>(*this);
    }
    DEFINE_MATCH_METHOD();
  };

  template <typename Arg0, typename... Args>
  AnfExpr Call(const LetVar& f, Arg0 arg0, Args&&... args) {
    return Call(f.name(),
                std::vector<CallArg>{std::forward<Arg0>(arg0),
                                     std::forward<Args>(args)...});
  }

  AnfExpr Call(const LetVar& f) {
    return Call(f.name(), std::vector<CallArg>{});
  }

  AnfExpr Call(const LetVar& f, const std::vector<LetVar>& vars) {
    return Call(f.name(), vars);
  }

  AnfExpr Call(const std::string& f) {
    return CallImpl(f, std::vector<CallArg>{});
  }

  template <typename Arg0, typename... Args>
  AnfExpr Call(const std::string& f, Arg0 arg0, Args&&... args) {
    return CallImpl(f,
                    std::vector<CallArg>{std::forward<Arg0>(arg0),
                                         std::forward<Args>(args)...});
  }

  AnfExpr Call(const std::string& f, const std::vector<LetVar>& vars) {
    std::vector<CallArg> args;
    args.reserve(vars.size());
    for (const auto& var : vars) {
      args.emplace_back(var);
    }
    return CallImpl(f, args);
  }

  AnfExpr Call(const std::string& f, const std::vector<AnfExpr>& vars) {
    std::vector<CallArg> args;
    args.reserve(vars.size());
    for (const auto& var : vars) {
      args.emplace_back(var);
    }
    return CallImpl(f, args);
  }

  const std::vector<Bind<AnfExpr>>& bindings() { return bindings_; }

  std::string NewTmpVarName() {
    static const std::string prefix = "__lambda_expr_tmp";
    return prefix + std::to_string(SeqNoGenerator_());
  }

 private:
  friend class LetVar;

  AnfExpr CallImpl(const std::string& f, const std::vector<CallArg>& args) {
    std::vector<Atomic<AnfExpr>> atomic_args;
    atomic_args.reserve(args.size());
    for (const auto& arg : args) {
      arg.Match(
          [&](const LetVar& var) {
            atomic_args.push_back(tVar<std::string>{var.name()});
          },
          [&](const AnfExpr& anf_expr) {
            arg.Match(
                [&](const Atomic<AnfExpr>& atomic) {
                  atomic_args.push_back(atomic);
                },
                [&](const auto&) {
                  atomic_args.push_back(BindToTmpVar(anf_expr));
                });
          });
    }
    return AnfExprBuilder().Call(tVar<std::string>{f}, atomic_args);
  }

  tVar<std::string> BindToTmpVar(const AnfExpr& anf_val) {
    const tVar<std::string> tmp_var_name{NewTmpVarName()};
    AddBinding(tmp_var_name.value(), anf_val);
    return tmp_var_name;
  }

  void AddBinding(const std::string& name, const AnfExpr& anf_val) {
    AnfExprBuilder anf;
    anf_val.Match(
        [&](const Atomic<AnfExpr>& atomic) {
          const auto& combined =
              anf.Call(tVar<std::string>{kBuiltinIdentity()}, {atomic});
          bindings_.push_back(anf.Bind(name, combined));
        },
        [&](const Combined<AnfExpr>& combined) {
          bindings_.push_back(anf.Bind(name, combined));
        },
        [&](const Let<AnfExpr>& let) {
          const auto& lambda = anf.Lambda({}, let);
          const auto& combined = anf.Call(lambda, {});
          bindings_.push_back(anf.Bind(name, combined));
        });
  }

  std::unordered_map<std::string, std::unique_ptr<LetVar>> let_var_storage_;
  std::vector<Bind<AnfExpr>> bindings_;
  std::function<size_t()> SeqNoGenerator_;
};

inline LetVar& LetVar::operator=(const LetVar& let_var) {
  AnfExprBuilder anf{};
  return *this = anf.Call(tVar<std::string>{kBuiltinIdentity()},
                          {tVar<std::string>{let_var.name()}});
}

inline LetVar& LetVar::operator=(const AnfExpr& anf_val) {
  let_ctx_->AddBinding(name_, anf_val);
  return *this;
}

inline LetVar& LetVar::Attr(const std::string& atttr_name) {
  AnfExprBuilder anf{};
  AnfExpr anf_expr =
      anf.Call(tVar<std::string>{kBuiltinGetAttr()},
               {tVar<std::string>{name()}, anf.String(atttr_name)});
  return let_ctx_->Var(let_ctx_->BindToTmpVar(anf_expr).value());
}

inline LetVar& LetVar::At(int64_t idx) {
  AnfExprBuilder anf{};
  AnfExpr anf_expr = anf.Call(tVar<std::string>{kBuiltinGetItem()},
                              {tVar<std::string>{name()}, anf.Int64(idx)});
  return let_ctx_->Var(let_ctx_->BindToTmpVar(anf_expr).value());
}

template <typename... Args>
inline AnfExpr LetVar::Call(Args&&... args) {
  return let_ctx_->Call(*this, std::forward<Args>(args)...);
}

class LambdaExprBuilder {
 public:
  LambdaExprBuilder() : SeqNoGenerator_(&LambdaExprBuilder::GenSeqNo) {}
  explicit LambdaExprBuilder(const std::function<size_t()>& SeqNoGenerator)
      : SeqNoGenerator_(SeqNoGenerator) {}
  LambdaExprBuilder(const LambdaExprBuilder&) = delete;
  LambdaExprBuilder(LambdaExprBuilder&&) = delete;

  AnfExpr Lambda(const std::vector<std::string>& args,
                 const std::function<LetVar(LetContext&)>& GetBody) {
    std::function<AnfExpr(LetContext&)> GetAnfExprBody =
        [&](LetContext& ctx) -> AnfExpr {
      return Atomic<AnfExpr>{tVar<std::string>{GetBody(ctx).name()}};
    };
    return Lambda(args, GetAnfExprBody);
  }

  AnfExpr Lambda(const std::vector<std::string>& args,
                 const std::function<AnfExpr(LetContext&)>& GetBody) {
    AnfExpr anf_expr = Let(GetBody);
    AnfExpr lambda_or_body = anf_expr.Match(
        [&](const ap::axpr::Let<AnfExpr>& let) {
          if (let->bindings.empty()) {
            return let->body;
          } else {
            return anf_expr;
          }
        },
        [&](const auto&) { return anf_expr; });
    return anf_.Lambda(MakeLambdaArgs(args), lambda_or_body);
  }

  AnfExpr Let(const std::function<AnfExpr(LetContext&)>& GetBody) {
    LetContext let_ctx{SeqNoGenerator_};
    AnfExpr ret = GetBody(let_ctx);
    return anf_.Let(let_ctx.bindings(), ret);
  }

  adt::Result<AnfExpr> Let(
      const std::function<adt::Result<AnfExpr>(LetContext&)>& GetBody) {
    LetContext let_ctx{SeqNoGenerator_};
    ADT_LET_CONST_REF(ret, GetBody(let_ctx));
    return anf_.Let(let_ctx.bindings(), ret);
  }

  std::vector<tVar<std::string>> MakeLambdaArgs(
      const std::vector<std::string>& args) {
    std::vector<tVar<std::string>> lambda_args;
    lambda_args.reserve(args.size());
    for (const auto& arg : args) {
      lambda_args.emplace_back(arg);
    }
    return lambda_args;
  }

 private:
  static size_t GenSeqNo() {
    static std::atomic<int64_t> seq_no(0);
    return seq_no++;
  }

  std::function<size_t()> SeqNoGenerator_;
  AnfExprBuilder anf_;
};

}  // namespace ap::axpr
