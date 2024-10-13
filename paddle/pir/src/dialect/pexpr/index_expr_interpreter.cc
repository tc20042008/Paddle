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
#include "ap/index_expr/index_expr_interpreter.h"
#include "ap/axpr/cps_expr_interpreter.h"
#include "ap/index_expr/index_expr_builtin_functions.h"

namespace ap::index_expr {

using axpr::CoreExpr;
using axpr::Lambda;

class IndexExprInterpreterImpl : public axpr::CpsExprInterpreter<Val> {
 public:
  explicit IndexExprInterpreterImpl(const std::shared_ptr<EnvMgr>& env_mgr)
      : axpr::CpsExprInterpreter<Val>(env_mgr,
                                      axpr::Frame<Val>{InitBuiltins()}) {}
  IndexExprInterpreterImpl(const IndexExprInterpreterImpl&) = delete;
  IndexExprInterpreterImpl(IndexExprInterpreterImpl&&) = delete;

 private:
  static axpr::Object<Val> InitBuiltins() {
    return axpr::Object<Val>{std::unordered_map<std::string, Val>{
        {"kUndefinedIndexTupleExpr",
         Val{IndexTupleExpr{UndefinedIndexTupleExpr{}}}},
        {"kNothingIndexTupleExpr",
         Val{IndexTupleExpr{NothingIndexTupleExpr{}}}},
        {"kIntArrayLikeIndexTupleExpr",
         Val{IndexTupleExpr{IntArrayLikeIndexTupleExpr{}}}},
        {"kUndefinedIndexExpr", Val{IndexExpr{UndefinedIndexExpr{}}}},
        {"PtrGetItem", Val{&MakePtrGetItem<Val>}},
        {"IndexExprBroadcastMask", Val{&MakeIndexExprBroadcastMask<Val>}},
        {"Slice", Val{&MakeSlice<Val>}},
        {"IndexExprSlice", Val{&MakeIndexExprSlice<Val>}},
        {"IndexExprAffine", Val{&MakeIndexExprAffine<Val>}},
        {"DisjointUnion", Val{&MakeDisjointUnion<Val>}},
        {"IndexTupleExprPermute", Val{&MakeIndexTupleExprPermute<Val>}},
        {"IndexTupleExprReshape", Val{&MakeIndexTupleExprReshape<Val>}},
        {"IndexTupleExprTransform", Val{&MakeIndexTupleExprTransform<Val>}},
        {"OpIndexTupleExprSignature", Val{&MakeOpIndexTupleExprSignature<Val>}},
        {"InIndexTupleExprSignature", Val{&MakeInIndexTupleExprSignature<Val>}},
        {"OutIndexTupleExprSignature",
         Val{&MakeOutIndexTupleExprSignature<Val>}},
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
    const axpr::Lambda<axpr::CoreExpr>& lambda,
    const std::vector<Val>& args) const {
  const auto& env = env_mgr_->New(impl_->builtin_env());
  axpr::Closure<Val> closure{lambda, env};
  adt::Result<Val> ret = impl_->Interpret(closure, args);
  env_mgr_->ClearAllFrames();
  return ret;
}

Result<Val> IndexExprInterpreter::operator()(
    const std::unordered_map<std::string, axpr::BuiltinFuncType<Val>>&
        global_functions,
    const axpr::Lambda<axpr::CoreExpr>& lambda,
    const std::vector<Val>& args) const {
  const auto& env = env_mgr_->New(impl_->builtin_env());
  for (const auto& [name, val] : global_functions) {
    env->Set(name, Val{val});
  }
  axpr::Closure<Val> closure{lambda, env};
  adt::Result<Val> ret = impl_->Interpret(closure, args);
  env_mgr_->ClearAllFrames();
  return ret;
}

}  // namespace ap::index_expr
