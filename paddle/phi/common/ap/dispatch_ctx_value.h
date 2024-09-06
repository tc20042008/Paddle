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
#include "paddle/phi/common/ap/adt.h"
#include "paddle/phi/common/ap/arg_value.h"
#include "paddle/phi/common/ap/const_tensor.h"
#include "paddle/phi/common/ap/data_type.h"
#include "paddle/phi/common/ap/define_ctx_value.h"
#include "paddle/phi/common/ap/dispatch_ctx.h"
#include "paddle/phi/common/ap/dispatch_raw_ctx.h"
#include "paddle/phi/common/ap/mutable_tensor.h"
#include "paddle/phi/common/ap/typed_buffer.h"
#include "paddle/pir/include/dialect/pexpr/value.h"

namespace ap::kernel_dispatch {

template <typename ValueT>
using ValueImpl = pexpr::ValueBase<ValueT,
                                   ConstTensor<ValueT>,
                                   MutableTensor<ValueT>,
                                   DispatchRawCtx<ValueT>,
                                   DispatchCtx<ValueT>>;

struct Value : public ValueImpl<Value> {
  using ValueImpl<Value>::ValueImpl;
  DEFINE_ADT_VARIANT_METHODS(ValueImpl<Value>);
};

using Val = Value;

using Env = pexpr::Environment<Val>;

using EnvMgr = pexpr::EnvironmentManager<Val>;

}  // namespace ap::kernel_dispatch
