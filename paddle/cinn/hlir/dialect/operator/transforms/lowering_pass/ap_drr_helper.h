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
#include "ap/drr/drr_value.h"

namespace cinn::dialect::ir {

struct ApDrrHelper {
  using CoreExpr = ap::axpr::CoreExpr;
  using Lambda = ap::axpr::Lambda<CoreExpr>;

  using DrrValue = ap::drr::Value;
  using DrrNode = ap::drr::Node<DrrValue>;
  using DrrCtx = ap::drr::DrrCtx<DrrValue, DrrNode>;

  adt::Result<DrrCtx> Interpret(const Lambda& lambda,
                                const std::string& drr_pass_name);
};

}  // namespace cinn::dialect::ir
