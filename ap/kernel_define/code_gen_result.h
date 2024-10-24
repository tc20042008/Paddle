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

#include "ap/axpr/adt.h"
#include "ap/axpr/core_expr.h"
#include "ap/axpr/type.h"
#include "ap/kernel_define/module.h"

namespace ap::kernel_define {

struct CodeGenResultImpl {
  Module code_module;
};

DEFINE_ADT_RC(CodeGenResult, CodeGenResultImpl);

}  // namespace ap::kernel_define

namespace ap::axpr {

template <>
struct TypeImpl<kernel_define::CodeGenResult> : public std::monostate {
  using value_type = kernel_define::CodeGenResult;

  const char* Name() const { return "CodeGenResult"; }
};

}  // namespace ap::axpr
