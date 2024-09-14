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
#include "ap/kernel/arg_value.h"
#include "ap/kernel/const_tensor.h"
#include "ap/kernel/data_type.h"
#include "ap/kernel/define_ctx_value.h"
#include "ap/kernel/dispatch_ctx.h"
#include "ap/kernel/dispatch_raw_ctx.h"
#include "ap/kernel/mutable_tensor.h"
#include "ap/kernel/typed_buffer.h"

namespace ap::kernel_dispatch {

template <typename ValueT>
using ValueImpl = ap::axpr::ValueBase<ValueT,
                                      ap::axpr::DataType,
                                      ap::axpr::DataValue,
                                      ap::axpr::PointerType,
                                      ap::axpr::PointerValue,
                                      ap::axpr::Object<ValueT>,
                                      ConstTensor<ValueT>,
                                      MutableTensor<ValueT>,
                                      DispatchRawCtx<ValueT>,
                                      DispatchCtx<ValueT>>;

struct Value : public ValueImpl<Value> {
  using ValueImpl<Value>::ValueImpl;
  DEFINE_ADT_VARIANT_METHODS(ValueImpl<Value>);
};

using Val = Value;

using Env = ap::axpr::Environment<Val>;

using EnvMgr = ap::axpr::EnvironmentManager<Val>;

}  // namespace ap::kernel_dispatch
