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

namespace ap::code_gen {

template <typename BirNode>
using IrOpImpl = std::variant<typename BirNode::native_op_type,
                              typename BirNode::packed_op_type,
                              typename BirNode::ref_op_type>;

template <typename BirNode>
struct IrOp : public IrOpImpl<BirNode> {
  using IrOpImpl<BirNode>::IrOpImpl;
  DEFINE_ADT_VARIANT_METHODS(IrOpImpl<BirNode>);

  template <typename ValueT>
  static adt::Result<IrOp> CastFrom(const ValueT& val) {
    return val.Match(
        [](const typename BirNode::native_op_type& impl) -> adt::Result<IrOp> {
          return impl;
        },
        [](const typename BirNode::packed_op_type& impl) -> adt::Result<IrOp> {
          return impl;
        },
        [](const typename BirNode::ref_op_type& impl) -> adt::Result<IrOp> {
          return impl;
        },
        [](const auto&) -> adt::Result<IrOp> {
          return adt::errors::ValueError{"IrOp::CastFrom failed."};
        });
  }
};

}  // namespace ap::code_gen
