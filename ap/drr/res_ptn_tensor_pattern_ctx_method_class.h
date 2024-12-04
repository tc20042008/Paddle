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
#include "ap/drr/drr_value.h"
#include "ap/drr/drr_value_helper.h"
#include "ap/drr/source_pattern_ctx.h"
#include "ap/drr/tags.h"
#include "ap/drr/tensor_pattern_ctx.h"

namespace ap::drr {

struct ResPtnTensorPatternCtx {
  using This = ResPtnTensorPatternCtx;
  using ObjT = drr::tResPtn<drr::TensorPatternCtx>;
  using Self = ObjT;
  using Helper = OpTensorPatternCtxHelper;

  static adt::Result<axpr::Value> ToString(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    std::ostringstream ss;
    const void* ptr = self.value().__adt_rc_shared_ptr_raw_ptr();
    ss << "<" << drr::Type<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  static adt::Result<axpr::Value> Hash(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const void* ptr = self.value().__adt_rc_shared_ptr_raw_ptr();
    return reinterpret_cast<int64_t>(ptr);
  }

  static adt::Result<axpr::Value> GetAttr(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);
    const auto& arg = args.at(0);
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_LET_CONST_REF(tensor_name, arg.template CastTo<std::string>());
    const auto& opt_ir_value =
        Helper{}.GetIrValueByUid(self.value(), tensor_name);
    if (opt_ir_value.HasOkValue()) {
      const auto& drr_value = opt_ir_value.GetOkValue().Match(
          [](const auto& impl) -> DrrValue { return ResPtn(impl); });
      return DrrValueHelper{}.CastToAxprValue(drr_value);
    }
    ADT_LET_CONST_REF(drr_ctx_ptr, adt::WeakPtrLock(self.value()->drr_ctx));
    const auto& src_tensor_ctx =
        drr_ctx_ptr->source_pattern_ctx.value()->tensor_pattern_ctx;
    ADT_LET_CONST_REF(src_ir_value,
                      Helper{}.GetIrValueByUid(src_tensor_ctx, tensor_name))
        << adt::errors::AttributeError{
               std::string() + "no source pattern binding tensor named '" +
               tensor_name + "' found."};
    const auto& match_result = src_ir_value.Match(
        [&](const NativeIrValue<drr::Node>& impl) -> adt::Result<DrrValue> {
          ADT_LET_CONST_REF(
              cloned, Helper{}.CloneIrValueDataAndRegister(self.value(), impl));
          return ResPtn(cloned);
        },
        [&](const PackedIrValue<drr::Node>& impl) -> adt::Result<DrrValue> {
          ADT_LET_CONST_REF(
              cloned, Helper{}.CloneIrValueDataAndRegister(self.value(), impl));
          return ResPtn(cloned);
        },
        [&](const auto&) -> adt::Result<axpr::Value> {
          return adt::errors::AttributeError{
              std::string() + "no source pattern binding tensor named '" +
              tensor_name + "' found."};
        });
    ADT_LET_CONST_REF(drr_value, match_result);
    return DrrValueHelper{}.CastToAxprValue(drr_value);
  }
};

inline const axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>&
GetResPtnTensorPatternCtxClass() {
  using ClassT = axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>;
  using Impl = drr::ResPtnTensorPatternCtx;
  using TT = drr::Type<drr::tResPtn<drr::TensorPatternCtx>>;
  static ClassT cls(
      axpr::MakeBuiltinClass<axpr::Value>(TT{}.Name(), [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
        Define("__getattr__", &Impl::GetAttr);
      }));
  return cls;
}

}  // namespace ap::drr
