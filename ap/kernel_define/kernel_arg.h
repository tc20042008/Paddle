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
#include "ap/axpr/core_expr.h"
#include "ap/axpr/object.h"
#include "ap/axpr/type.h"
#include "ap/kernel_define/adt.h"
#include "ap/kernel_define/arg_type.h"

namespace ap::kernel_define {

struct KernelArgImpl {
  ArgType arg_type;
  axpr::Lambda<axpr::CoreExpr> getter_lambda;

  bool operator==(const KernelArgImpl& other) const {
    return this->arg_type == other.arg_type &&
           this->getter_lambda == other.getter_lambda;
  }
};

DEFINE_ADT_RC(KernelArg, KernelArgImpl);

}  // namespace ap::kernel_define
