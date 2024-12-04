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
#include "ap/include/kernel_dispatch/value.h"

namespace phi {

struct KernelDispatchHelper {
  using CoreExpr = ap::axpr::CoreExpr;
  using Lambda = ap::axpr::Lambda<CoreExpr>;
  using Val = ap::kernel_dispatch::Val;
  using DispatchCtx = ap::kernel_dispatch::DispatchCtx<Val>;

  adt::Result<Val> InterpretCtxMaker(const Lambda& ctx_maker_lambda);

  adt::Result<adt::Ok> InterpretKernelDispatcher(
      const Lambda& kernel_dispatch_lambda, const DispatchCtx& dispatch_ctx);
};

}  // namespace phi
