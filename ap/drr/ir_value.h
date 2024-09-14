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
#include "ap/drr/native_ir_value.h"
#include "ap/drr/packed_ir_value.h"

namespace ap::drr {

template <typename NodeT>
using IrValueImpl = std::variant<NativeIrValue<NodeT>, PackedIrValue<NodeT>>;

template <typename NodeT>
struct IrValue : public IrValueImpl<NodeT> {
  using IrValueImpl<NodeT>::IrValueImpl;
  DEFINE_ADT_VARIANT_METHODS(IrValueImpl<NodeT>);

  const graph::Node<NodeT>& node() const {
    return Match([](const auto& impl) -> const graph::Node<NodeT>& {
      return impl->node;
    });
  }

  const std::string& name() const {
    return Match(
        [](const auto& impl) -> const std::string& { return impl->name; });
  }
};

}  // namespace ap::drr
