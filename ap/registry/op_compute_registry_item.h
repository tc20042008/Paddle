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

#include <vector>
#include "ap/adt/adt.h"
#include "ap/axpr/core_expr.h"
#include "ap/registry/cell.h"
#include "ap/registry/nice.h"

namespace ap::registry {

struct OpComputeRegistryItemImpl {
  std::string op_name;
  std::string arch_type;
  Nice nice;
  Cell<axpr::Lambda<axpr::CoreExpr>> lambda;
};

DEFINE_ADT_RC(OpComputeRegistryItem, OpComputeRegistryItemImpl);

}  // namespace ap::registry
