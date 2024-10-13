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
#include "ap/registry/cell.h"

namespace ap::registry {

struct SetterDecoratorImpl {
  Cell<axpr::Lambda<axpr::CoreExpr>> lambda;

  bool operator==(const SetterDecoratorImpl& other) const {
    return this == &other;
  }
};

DEFINE_ADT_RC(SetterDecorator, SetterDecoratorImpl);

}  // namespace ap::registry

namespace ap::axpr {

template <>
struct TypeImpl<registry::SetterDecorator> : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "SetterDecorator"; }
};

}  // namespace ap::axpr
