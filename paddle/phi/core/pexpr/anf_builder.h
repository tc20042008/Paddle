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
#include "paddle/phi/core/pexpr/atomic_builder.h"

namespace pexpr {

class AnfExprBuilder : public AtomicExprBuilder<AnfExpr> {
 public:
  AnfExprBuilder() {}
  AnfExprBuilder(const AnfExprBuilder&) = delete;
  AnfExprBuilder(AnfExprBuilder&&) = delete;

  Combined<AnfExpr> Call(const Atomic<AnfExpr>& f,
                         const std::vector<Atomic<AnfExpr>>& args) {
    return Combined<AnfExpr>{pexpr::Call<AnfExpr>{f, args}};
  }

  Combined<AnfExpr> If(const Atomic<AnfExpr>& c,
                       const AnfExpr& t,
                       const AnfExpr& f) {
    return Combined<AnfExpr>{pexpr::If<AnfExpr>{
        c, std::make_shared<AnfExpr>(t), std::make_shared<AnfExpr>(f)}};
  }

  pexpr::Bind<AnfExpr> Bind(const std::string& var,
                            const Combined<AnfExpr>& val) {
    return pexpr::Bind<AnfExpr>{tVar<std::string>{var}, val};
  }

  pexpr::Let<AnfExpr> Let(const std::vector<pexpr::Bind<AnfExpr>>& assigns,
                          const AnfExpr& body) {
    return pexpr::Let<AnfExpr>{assigns, std::make_shared<AnfExpr>(body)};
  }

  AnfExpr operator()(const Atomic<AnfExpr>& atomic) { return AnfExpr{atomic}; }

  AnfExpr operator()(const Combined<AnfExpr>& combined) {
    return AnfExpr{combined};
  }

  AnfExpr operator()(const pexpr::Let<AnfExpr>& let) { return AnfExpr{let}; }

 private:
};

}  // namespace pexpr
