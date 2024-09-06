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
#include "paddle/pir/include/dialect/pexpr/cps_expr_interpreter.h"
#include "paddle/pir/include/dialect/pexpr/index_expr_builtin_functions.h"

namespace pexpr::index_expr {

class IndexExprInterpreterImpl : public CpsExprInterpreter<Val> {
 public:
  explicit IndexExprInterpreterImpl(const std::shared_ptr<EnvMgr>& env_mgr)
      : CpsExprInterpreter<Val>(env_mgr, Frame<Val>{InitBuiltins()}) {}
  IndexExprInterpreterImpl(const IndexExprInterpreterImpl&) = delete;
  IndexExprInterpreterImpl(IndexExprInterpreterImpl&&) = delete;

 private:
  static Object<Val> InitBuiltins() {
    return Object<Val>{std::unordered_map<std::string, Val>{
        {"list", Val{&BuiltinList<Val>}},
        {"kUndefinedIndexTupleExpr",
         Val{IndexTupleExpr{UndefinedIndexTupleExpr{}}}},
        {"kNothingIndexTupleExpr",
         Val{IndexTupleExpr{NothingIndexTupleExpr{}}}},
        {"kIntArrayLikeIndexTupleExpr",
         Val{IndexTupleExpr{IntArrayLikeIndexTupleExpr{}}}},
        {"kUndefinedIndexExpr", Val{IndexExpr{UndefinedIndexExpr{}}}},
        {"PtrGetItem", Val{&MakePtrGetItem}},
        {"IndexExprBroadcastMask", Val{&MakeIndexExprBroadcastMask}},
        {"Slice", Val{&MakeSlice}},
        {"IndexExprSlice", Val{&MakeIndexExprSlice}},
        {"IndexExprAffine", Val{&MakeIndexExprAffine}},
        {"DisjointUnion", Val{&MakeDisjointUnion}},
        {"IndexTupleExprPermute", Val{&MakeIndexTupleExprPermute}},
        {"IndexTupleExprReshape", Val{&MakeIndexTupleExprReshape}},
        {"IndexTupleExprTransform", Val{&MakeIndexTupleExprTransform}},
        {"OpIndexTupleExprSignature", Val{&MakeOpIndexTupleExprSignature}},
        {"InIndexTupleExprSignature", Val{&MakeInIndexTupleExprSignature}},
        {"OutIndexTupleExprSignature", Val{&MakeOutIndexTupleExprSignature}},
    }};
  }
};

IndexExprInterpreter::IndexExprInterpreter()
    : env_mgr_(std::make_shared<EnvMgr>()),
      impl_(std::make_unique<IndexExprInterpreterImpl>(env_mgr_)) {}

IndexExprInterpreter::IndexExprInterpreter(
    const std::shared_ptr<EnvMgr>& env_mgr)
    : env_mgr_(env_mgr),
      impl_(std::make_unique<IndexExprInterpreterImpl>(env_mgr)) {}

Result<Val> IndexExprInterpreter::operator()(
    const Lambda<CoreExpr>& lambda, const std::vector<Val>& args) const {
  const auto& env = env_mgr_->New(impl_->builtin_env());
  Closure<Val> closure{lambda, env};
  Result<Val> ret = impl_->Interpret(closure, args);
  env_mgr_->ClearAllFrames();
  return ret;
}

Result<Val> IndexExprInterpreter::operator()(
    const std::unordered_map<std::string, BuiltinFuncType<Val>>&
        global_functions,
    const Lambda<CoreExpr>& lambda,
    const std::vector<Val>& args) const {
  const auto& env = env_mgr_->New(impl_->builtin_env());
  for (const auto& [name, val] : global_functions) {
    env->Set(name, Val{val});
  }
  Closure<Val> closure{lambda, env};
  Result<Val> ret = impl_->Interpret(closure, args);
  env_mgr_->ClearAllFrames();
  return ret;
}

}  // namespace pexpr::index_expr
