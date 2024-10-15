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
#include "ap/kernel_define/define_ctx.h"
#include "ap/kernel_define/module.h"
#include "ap/paddle/pir_node.h"

namespace cinn::dialect::ir {

struct ApKernelDefineHelper {
  using CoreExpr = ap::axpr::CoreExpr;
  using Lambda = ap::axpr::Lambda<CoreExpr>;
  using Module = ap::kernel_define::Module;
  using PirNode = ap::paddle::PirNode;
  using DefineCtx = ap::kernel_define::DefineCtx<PirNode>;

  adt::Result<Module> Interpret(const Lambda& lambda,
                                const DefineCtx& define_ctx);
};

}  // namespace cinn::dialect::ir
