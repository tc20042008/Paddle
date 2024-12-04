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

#include <unordered_set>
#include "ap/include/axpr/method_class.h"
#include "ap/include/axpr/type.h"
#include "ap/include/drr/drr_value.h"
#include "ap/include/drr/native_ir_value.h"
#include "ap/include/drr/op_tensor_pattern_ctx_helper.h"
#include "ap/include/drr/tags.h"
#include "ap/include/drr/unbound_ir_value.h"
#include "ap/include/drr/unbound_native_ir_op.h"

namespace ap::drr {

struct ResPtnUnboundNativeIrOpMethodClass {
  using This = ResPtnUnboundNativeIrOpMethodClass;
  using Self = tResPtn<UnboundNativeIrOp<drr::Node>>;

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

  using Helper = OpTensorPatternCtxHelper;

  static adt::Result<axpr::Value> StaticCall(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return This{}.Call(self, args);
  }

  adt::Result<axpr::Value> Call(const Self& self,
                                const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 2) << adt::errors::TypeError{
        std::string() +
        "ResPtnUnboundNativeIrOp.__call__ takes 2 arguments. but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(input_vals,
                      args.at(0).template CastTo<adt::List<axpr::Value>>())
        << adt::errors::TypeError{
               std::string() +
               "the first argument of ResPtnUnboundNativeIrOp.__call__ should "
               "be a list."};
    adt::List<NativeIrValue<drr::Node>> inputs;
    inputs->reserve(input_vals->size());
    for (const auto& input_val : *input_vals) {
      ADT_LET_CONST_REF(
          input, input_val.template CastTo<tResPtn<NativeIrValue<drr::Node>>>())
          << adt::errors::TypeError{
                 std::string() +
                 "unsupported operand types for "
                 "ResPtnUnboundNativeIrOp.__call__ inputs: '" +
                 axpr::GetTypeName(input_val) + "'."};
      inputs->emplace_back(input.value());
    }
    ADT_LET_CONST_REF(output_vals,
                      args.at(1).template CastTo<adt::List<axpr::Value>>())
        << adt::errors::TypeError{
               std::string() +
               "the second argument of ResPtnUnboundNativeIrOp.__call__ should "
               "be a list."};
    adt::List<NativeIrValue<drr::Node>> outputs;
    outputs->reserve(output_vals->size());
    for (const auto& output_val : *output_vals) {
      ADT_LET_CONST_REF(
          output,
          output_val.template CastTo<tResPtn<NativeIrValue<drr::Node>>>())
          << adt::errors::TypeError{
                 std::string() +
                 "unsupported operand types for "
                 "ResPtnUnboundNativeIrOp.__call__ outputs: '" +
                 axpr::GetTypeName(output_val) + "'."};
      outputs->emplace_back(output.value());
    }
    ADT_RETURN_IF_ERR(CheckNoRedundentTensorNames(inputs, outputs));
    ADT_LET_CONST_REF(native_op,
                      Helper{}.GetNativeIrOpByUnboundNativeIrOp(self.value()));
    Helper{}.ConnectIrOpAndIrValue(native_op, inputs, outputs);
    return adt::Nothing{};
  }

  adt::Result<adt::Ok> CheckNoRedundentTensorNames(
      const adt::List<NativeIrValue<drr::Node>>& inputs,
      const adt::List<NativeIrValue<drr::Node>>& outputs) {
    std::unordered_set<std::string> existed_names;
    for (const auto& input : *inputs) {
      existed_names.insert(input->name);
    }
    for (const auto& output : *outputs) {
      ADT_CHECK(existed_names.emplace(output->name).second)
          << adt::errors::TypeError{std::string() + "redundant tensor name '" +
                                    output->name + "' detected."};
    }
    return adt::Ok{};
  }
};

inline const axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>&
GetResPtnUnboundNativeIrOpClass() {
  using ClassT = axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>;
  using TT = drr::Type<drr::tResPtn<drr::UnboundNativeIrOp<drr::Node>>>;
  using Impl = ResPtnUnboundNativeIrOpMethodClass;
  static ClassT cls(
      axpr::MakeBuiltinClass<axpr::Value>(TT{}.Name(), [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
        Define("__call__", &Impl::StaticCall);
      }));
  return cls;
}

}  // namespace ap::drr
