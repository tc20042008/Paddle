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
#include <cstdint>
#include "paddle/phi/common/ap/define_ctx_value.h"
#include "paddle/pir/include/dialect/pexpr/core_expr_interpreter.h"

namespace ap::kernel_define {

class KernelDefinerInterpreter : public pexpr::CoreExprInterpreter<Val> {
 public:
  KernelDefinerInterpreter()
      : env_mgr_(std::make_shared<EnvMgr>()),
        CoreExprInterpreter<Val>(env_mgr_.get(), MakeBuiltinFrame()) {}
  KernelDefinerInterpreter(const KernelDefinerInterpreter&) = delete;
  KernelDefinerInterpreter(KernelDefinerInterpreter&&) = delete;

  adt::Result<Val> operator()(const pexpr::Lambda<pexpr::CoreExpr>& lambda,
                              const Val& ctx) {
    const auto& env = env_mgr_->New(this->builtin_frame());
    pexpr::NaiveClosure<Val> closure{lambda, env};
    adt::Result<Val> ret = InterpretClosure(closure, {ctx});
    env_mgr_->ClearAllFrames();
    return ret;
  }

 private:
  static pexpr::Frame<Val> MakeBuiltinFrame() {
    return pexpr::Frame<Val>{InitBuiltins()};
  }
  static pexpr::Object<Val> InitBuiltins() {
    return pexpr::Object<Val>{std::unordered_map<std::string, Val>{}};
  }

 private:
  std::shared_ptr<EnvMgr> env_mgr_;
};

}  // namespace ap::kernel_define
