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
#include "ap/axpr/type.h"
#include "ap/kernel_define/adt.h"
#include "ap/kernel_define/arg_type.h"
#include "ap/kernel_define/data_type.h"
#include "ap/kernel_define/func_declare.h"
#include "ap/kernel_define/source_code.h"
#include "paddle/cinn/adt/adt.h"

namespace ap::kernel_define {

struct ModuleImpl {
  adt::List<FuncDeclare> func_declares;
  SourceCode source_code;

  bool operator==(const ModuleImpl& other) const {
    return other.func_declares == this->func_declares &&
           other.source_code == this->source_code;
  }
};
DEFINE_ADT_RC(Module, ModuleImpl);

}  // namespace ap::kernel_define

namespace ap::axpr {

template <>
struct TypeImpl<ap::kernel_define::Module> : public std::monostate {
  using value_type = ap::kernel_define::Module;

  const char* Name() const { return "Module"; }
};

}  // namespace ap::axpr
