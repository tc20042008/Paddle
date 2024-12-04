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
#include "ap/include/adt/adt.h"
#include "ap/include/axpr/function.h"
#include "ap/include/axpr/serializable_value.h"

namespace ap::registry {

struct DrrPassRegistryItemImpl {
  std::string drr_pass_name;
  int64_t nice;
  axpr::ClassAttrs<axpr::SerializableValue> cls;
};

DEFINE_ADT_RC(DrrPassRegistryItem, DrrPassRegistryItemImpl);

}  // namespace ap::registry
