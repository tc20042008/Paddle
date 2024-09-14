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

#include "ap/adt/adt.h"
#include "ap/axpr/type.h"

namespace ap::drr {

template <typename ValueT, typename NodeT>
struct TensorPatternCtxImpl;

template <typename ValueT, typename NodeT>
struct UnboundIrValueImpl {
  std::string name;
  std::weak_ptr<TensorPatternCtxImpl<ValueT, NodeT>> tensor_pattern_ctx;

  bool operator==(const UnboundIrValueImpl& other) const {
    return this->name == other.name &&
           this->tensor_pattern_ctx.lock() == other.tensor_pattern_ctx.lock();
  }
};

template <typename ValueT, typename NodeT>
DEFINE_ADT_RC(UnboundIrValue, UnboundIrValueImpl<ValueT, NodeT>);

}  // namespace ap::drr

namespace ap::axpr {

template <typename ValueT, typename NodeT>
struct TypeImpl<drr::UnboundIrValue<ValueT, NodeT>> : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "UnboundIrValue"; }
};

}  // namespace ap::axpr
