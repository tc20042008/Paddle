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

#include <unordered_map>
#include <vector>
#include "ap/adt/adt.h"
#include "ap/axpr/object.h"
#include "ap/axpr/type.h"
#include "ap/registry/drr_registry_item.h"
#include "ap/registry/module_template_registry_item.h"
#include "ap/registry/nice.h"
#include "ap/registry/op_compute_registry_item.h"
#include "ap/registry/op_indexes_expr_registry_item.h"

namespace ap::registry {

struct RegistryImpl {
  Key2Nice2Items<OpIndexesExprRegistryItem> op_indexes_expr_registry_items;
  Key2Nice2Items<DrrRegistryItem> drr_registry_items;
  Key2Nice2Items<OpComputeRegistryItem> op_compute_registry_items;
  Key2Nice2Items<ModuleTemplateRegistryItem> module_template_registry_items;

  bool operator==(const RegistryImpl& other) const { return this == &other; }
};

DEFINE_ADT_RC(Registry, RegistryImpl);

}  // namespace ap::registry

namespace ap::axpr {

template <>
struct TypeImpl<registry::Registry> : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "Registry"; }
};

}  // namespace ap::axpr
