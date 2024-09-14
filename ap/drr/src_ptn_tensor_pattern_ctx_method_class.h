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

#include "ap/axpr/method_class.h"
#include "ap/axpr/type.h"
#include "ap/drr/ir_value.h"
#include "ap/drr/op_tensor_pattern_ctx_helper.h"
#include "ap/drr/tags.h"
#include "ap/drr/tensor_pattern_ctx.h"

namespace ap::drr {

template <typename ValueT, typename NodeT>
struct SrcPtnTensorPatternCtx {
  using This = SrcPtnTensorPatternCtx;
  using ObjT = drr::tSrcPtn<drr::TensorPatternCtx<ValueT, NodeT>>;
  using Self = ObjT;
  using Helper = OpTensorPatternCtxHelper<ValueT, NodeT>;

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& arg) {
    ADT_LET_CONST_REF(tensor_name, axpr::TryGetImpl<std::string>(arg));

    const auto& opt_ir_value =
        Helper{}.GetIrValueByUid(self.value(), tensor_name);
    if (opt_ir_value.HasError()) {
      return UnboundIrValue<ValueT, NodeT>{tensor_name,
                                           self.value().shared_ptr()};
    }
    const auto& ir_value = opt_ir_value.GetOkValue();
    return ir_value.Match(
        [](const auto& impl) -> ValueT { return SrcPtn(impl); });
  }
};

}  // namespace ap::drr

namespace ap::axpr {

template <typename ValueT, typename NodeT>
struct MethodClassImpl<ValueT,
                       drr::tSrcPtn<drr::TensorPatternCtx<ValueT, NodeT>>>
    : public drr::SrcPtnTensorPatternCtx<ValueT, NodeT> {};

template <typename ValueT, typename NodeT>
struct MethodClassImpl<
    ValueT,
    TypeImpl<drr::tSrcPtn<drr::TensorPatternCtx<ValueT, NodeT>>>> {};

}  // namespace ap::axpr
