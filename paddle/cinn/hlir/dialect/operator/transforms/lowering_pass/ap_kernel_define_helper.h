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

#include "ap/adt/adt.h"
#include "ap/code_gen/code_gen_ctx.h"
#include "ap/code_gen/code_gen_result.h"
#include "ap/code_gen/value.h"
#include "ap/code_module/module.h"
#include "ap/paddle/pir_node.h"

namespace cinn::dialect::ir {

struct ApKernelDefineHelper {
  using CoreExpr = ap::axpr::CoreExpr;
  using Lambda = ap::axpr::Lambda<CoreExpr>;
  using Module = ap::code_module::Module;
  using PirNode = ap::paddle::PirNode;
  using CGValue = ap::code_gen::Value<PirNode>;
  using CodeGenCtx = ap::code_gen::CodeGenCtx<PirNode>;
  using CodeGenResult = ap::code_gen::CodeGenResult<CGValue>;

  adt::Result<CodeGenResult> Interpret(const Lambda& lambda,
                                       const CodeGenCtx& code_gen_ctx);
};

}  // namespace cinn::dialect::ir
