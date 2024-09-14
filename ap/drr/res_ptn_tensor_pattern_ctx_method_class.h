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
#include "ap/drr/source_pattern_ctx.h"
#include "ap/drr/tags.h"
#include "ap/drr/tensor_pattern_ctx.h"

namespace ap::drr {

template <typename ValueT, typename NodeT>
struct ResPtnTensorPatternCtx {
  using This = ResPtnTensorPatternCtx;
  using ObjT = drr::tResPtn<drr::TensorPatternCtx<ValueT, NodeT>>;
  using Self = ObjT;
  using Helper = OpTensorPatternCtxHelper<ValueT, NodeT>;

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& arg) {
    ADT_LET_CONST_REF(tensor_name, axpr::TryGetImpl<std::string>(arg));
    const auto& opt_ir_value =
        Helper{}.GetIrValueByUid(self.value(), tensor_name);
    if (opt_ir_value.HasOkValue()) {
      return opt_ir_value.GetOkValue().Match(
          [](const auto& impl) -> ValueT { return ResPtn(impl); });
    }
    ADT_LET_CONST_REF(drr_ctx_ptr, adt::WeakPtrLock(self.value()->drr_ctx));
    const auto& src_tensor_ctx =
        drr_ctx_ptr->source_pattern_ctx.value()->tensor_pattern_ctx;
    ADT_LET_CONST_REF(src_ir_value,
                      Helper{}.GetIrValueByUid(src_tensor_ctx, tensor_name))
        << adt::errors::AttributeError{
               std::string() + "no source pattern binding tensor named '" +
               tensor_name + "' found."};
    return src_ir_value.Match(
        [&](const NativeIrValue<NodeT>& impl) -> adt::Result<ValueT> {
          ADT_LET_CONST_REF(
              cloned, Helper{}.CloneIrValueDataAndRegister(self.value(), impl));
          return ResPtn(cloned);
        },
        [&](const PackedIrValue<NodeT>& impl) -> adt::Result<ValueT> {
          ADT_LET_CONST_REF(
              cloned, Helper{}.CloneIrValueDataAndRegister(self.value(), impl));
          return ResPtn(cloned);
        },
        [&](const auto&) -> adt::Result<ValueT> {
          return adt::errors::AttributeError{
              std::string() + "no source pattern binding tensor named '" +
              tensor_name + "' found."};
        });
  }
};

}  // namespace ap::drr

namespace ap::axpr {

template <typename ValueT, typename NodeT>
struct MethodClassImpl<ValueT,
                       drr::tResPtn<drr::TensorPatternCtx<ValueT, NodeT>>>
    : public drr::ResPtnTensorPatternCtx<ValueT, NodeT> {};

template <typename ValueT, typename NodeT>
struct MethodClassImpl<
    ValueT,
    TypeImpl<drr::tResPtn<drr::TensorPatternCtx<ValueT, NodeT>>>> {};

}  // namespace ap::axpr
