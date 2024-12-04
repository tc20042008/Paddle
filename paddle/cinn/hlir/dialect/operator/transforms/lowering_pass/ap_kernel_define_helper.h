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

#include "ap/include/adt/adt.h"
#include "ap/include/code_gen/code_gen_ctx.h"
#include "ap/include/code_gen/code_gen_result.h"
#include "ap/include/code_gen/value.h"
#include "ap/include/code_module/module.h"
#include "ap/include/paddle/pir_node.h"

namespace cinn::dialect::ir {

struct ApKernelDefineHelper {
  using Function = ap::axpr::Function<ap::axpr::SerializableValue>;
  using Module = ap::code_module::Module;
  using PirNode = ap::paddle::PirNode;
  using CGValue = ap::code_gen::Value;
  using CodeGenCtx = ap::code_gen::CodeGenCtx<PirNode>;
  using CodeGenResult = ap::code_gen::CodeGenResult<CGValue>;

  adt::Result<CodeGenResult> Interpret(const Function& lambda,
                                       const CodeGenCtx& code_gen_ctx);
};

}  // namespace cinn::dialect::ir
