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
#include "ap/ir_match/ir_match_ctx.h"

namespace ap::ir_match {

template <typename ValueT, typename IrNodeT>
struct IrMatchCtxMethodClass {
  using This = IrMatchCtxMethodClass;
  using Self = ir_match::IrMatchCtx<IrNodeT>;

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, axpr::TryGetImpl<std::string>(attr_name_val));
    if (attr_name == "op") {
      return GetOpMatchCtx(self);
    }
    if (attr_name == "tensor") {
      return GetTensorMatchCtx(self);
    }
    return adt::errors::TypeError{
        std::string() + "'IrMatchCtx' has no attribute '" + attr_name + "'"};
  }

  adt::Result<ValueT> GetOpMatchCtx(const Self& self) {
    auto* ir_match_ctx = self.shared_ptr().get();
    if (ir_match_ctx->op_match_ctx.has_value()) {
      return ir_match_ctx->op_match_ctx.value();
    }
    OpMatchCtx<IrNodeT> op_match_ctx{self.shared_ptr()};
    ir_match_ctx->op_match_ctx = op_match_ctx;
    return op_match_ctx;
  }

  adt::Result<ValueT> GetTensorMatchCtx(const Self& self) {
    auto* ir_match_ctx = self.shared_ptr().get();
    if (ir_match_ctx->tensor_match_ctx.has_value()) {
      return ir_match_ctx->tensor_match_ctx.value();
    }
    TensorMatchCtx<IrNodeT> tensor_match_ctx{self.shared_ptr()};
    ir_match_ctx->tensor_match_ctx = tensor_match_ctx;
    return tensor_match_ctx;
  }
};

}  // namespace ap::ir_match

namespace ap::axpr {

template <typename ValueT, typename IrNodeT>
struct MethodClassImpl<ValueT, ir_match::IrMatchCtx<IrNodeT>>
    : public ir_match::IrMatchCtxMethodClass<ValueT, IrNodeT> {};

template <typename ValueT, typename IrNodeT>
struct MethodClassImpl<ValueT, TypeImpl<ir_match::IrMatchCtx<IrNodeT>>> {};

}  // namespace ap::axpr
