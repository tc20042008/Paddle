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
#include "ap/axpr/value.h"
#include "ap/kernel/adt.h"
#include "ap/kernel/arg_type.h"
#include "ap/kernel/data_type.h"
#include "ap/kernel/dispatch_raw_ctx.h"
#include "ap/kernel/typed_buffer.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

class DenseTensor;

}

namespace ap::kernel_dispatch {

template <typename ValueT>
struct DispatchCtxImpl {
  DispatchRawCtx<ValueT> raw_ctx;
  ap::axpr::Object<ValueT> data;

  bool operator==(const DispatchCtxImpl& other) const { return &other == this; }
};

template <typename ValueT>
DEFINE_ADT_RC(DispatchCtx, DispatchCtxImpl<ValueT>);

}  // namespace ap::kernel_dispatch

namespace ap::axpr {

template <typename ValueT>
struct TypeImpl<ap::kernel_dispatch::DispatchCtx<ValueT>>
    : public std::monostate {
  using value_type = ap::kernel_dispatch::DispatchCtx<ValueT>;

  const char* Name() const { return "DispatchCtx"; }
};

}  // namespace ap::axpr
