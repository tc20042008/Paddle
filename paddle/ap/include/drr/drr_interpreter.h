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

#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/drr/value.h"
#include "paddle/ap/include/registry/abstract_drr_pass_registry_item.h"

namespace ap::drr {

struct DrrInterpreter {
  using Function = ap::axpr::Function<ap::axpr::SerializableValue>;

  using DrrNode = ap::drr::Node;
  using DrrCtx = ap::drr::DrrCtx;

  ap::adt::Result<DrrCtx> Interpret(
      const axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>&
          backend_ir_ctx,
      const Function& lambda,
      const std::string& abstract_drr_pass_name);
  ap::adt::Result<DrrCtx> Interpret(
      const axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>&
          backend_ir_ctx,
      const ap::axpr::ClassAttrs<ap::axpr::SerializableValue>& cls);
};

}  // namespace ap::drr
