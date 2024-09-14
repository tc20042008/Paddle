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
#include "ap/axpr/value.h"
#include "ap/kernel/adt.h"
#include "ap/kernel/data_type.h"
#include "ap/kernel/definer_ctx.h"
#include "ap/kernel/definer_raw_ctx.h"
#include "ap/kernel/func_declare.h"
#include "ap/kernel/module.h"
#include "ap/kernel/source_code.h"
#include "paddle/cinn/adt/adt.h"

namespace ap::kernel_define {

namespace adt = ::cinn::adt;

template <typename ValueT>
using ValueImpl = ap::axpr::ValueBase<ValueT,
                                      ap::axpr::DataType,
                                      ap::axpr::PointerType,
                                      ap::axpr::Object<ValueT>,
                                      DefinerRawCtx,
                                      DefinerCtx<ValueT>,
                                      FuncDeclare,
                                      SourceCode,
                                      Module>;

struct Value : public ValueImpl<Value> {
  using ValueImpl<Value>::ValueImpl;
  DEFINE_ADT_VARIANT_METHODS(ValueImpl<Value>);
};

using Val = Value;

using Env = ap::axpr::Environment<Val>;

using EnvMgr = ap::axpr::EnvironmentManager<Val>;

}  // namespace ap::kernel_define
