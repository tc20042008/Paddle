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
#include "paddle/pir/include/dialect/pexpr/core_expr_interpreter.h"
#include "paddle/pir/include/dialect/pexpr/index_expr_builtin_functions.h"

namespace pexpr::index_expr {

class IndexExprInterpreterImpl : public CoreExprInterpreter<Val> {
 public:
  explicit IndexExprInterpreterImpl(const std::shared_ptr<EnvMgr>& env_mgr)
      : CoreExprInterpreter<Val>(env_mgr, MakeBuiltinFrame()) {}
  IndexExprInterpreterImpl(const IndexExprInterpreterImpl&) = delete;
  IndexExprInterpreterImpl(IndexExprInterpreterImpl&&) = delete;

 private:
  static Frame<Val> MakeBuiltinFrame() { return Frame<Val>{InitBuiltins()}; }

  static Object<Val> InitBuiltins() {
    return Object<Val>{std::unordered_map<std::string, Val>{
        {kBuiltinIf, Val{BuiltinFuncType<Val>(&BuiltinIf<Val>)}},
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
  const auto& env = env_mgr_->New(impl_->builtin_frame());
  NaiveClosure<Val> closure{lambda, env};
  Result<Val> ret = (*impl_)(closure, args);
  env_mgr_->ClearAllFrames();
  return ret;
}

Result<Val> IndexExprInterpreter::operator()(
    const std::unordered_map<std::string, BuiltinFuncType<Val>>&
        global_functions,
    const Lambda<CoreExpr>& lambda,
    const std::vector<Val>& args) const {
  const auto& env = env_mgr_->New(impl_->builtin_frame());
  for (const auto& [name, val] : global_functions) {
    env->Set(name, Val{val});
  }
  NaiveClosure<Val> closure{lambda, env};
  Result<Val> ret = (*impl_)(closure, args);
  env_mgr_->ClearAllFrames();
  return ret;
}

}  // namespace pexpr::index_expr
