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
#include "ap/drr/native_ir_value.h"
#include "ap/drr/op_tensor_pattern_ctx_helper.h"
#include "ap/drr/tags.h"
#include "ap/drr/unbound_ir_value.h"
#include "ap/drr/unbound_native_ir_op.h"

namespace ap::drr {

template <typename ValueT, typename NodeT>
using SrcPtnValidIrValueImpl =
    std::variant<NativeIrValue<NodeT>, UnboundIrValue<ValueT, NodeT>>;

template <typename ValueT, typename NodeT>
struct SrcPtnValidIrValue : public SrcPtnValidIrValueImpl<ValueT, NodeT> {
  using SrcPtnValidIrValueImpl<ValueT, NodeT>::SrcPtnValidIrValueImpl;

  DEFINE_ADT_VARIANT_METHODS(SrcPtnValidIrValueImpl<ValueT, NodeT>);

  const std::string& name() const {
    return Match([](const auto& ir_value) -> const std::string& {
      return ir_value->name;
    });
  }
};

template <typename ValueT, typename NodeT>
struct SrcPtnUnboundNativeIrOp {
  using This = SrcPtnUnboundNativeIrOp;
  using Self = tSrcPtn<UnboundNativeIrOp<ValueT, NodeT>>;

  using Helper = OpTensorPatternCtxHelper<ValueT, NodeT>;

  adt::Result<ValueT> Call(const Self& self) {
    return axpr::Method<ValueT>(self, &This::StaticCall);
  }

  static adt::Result<ValueT> StaticCall(const ValueT& self_val,
                                        const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, axpr::TryGetImpl<Self>(self_val));
    return This{}.Call(self, args);
  }

  adt::Result<ValueT> Call(const Self& self, const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 2) << adt::errors::TypeError{
        std::string() +
        "SrcPtnUnboundNativeIrOp.__call__ takes 2 arguments. but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(input_vals,
                      axpr::TryGetImpl<adt::List<ValueT>>(args.at(0)))
        << adt::errors::TypeError{
               std::string() +
               "the first argument of SrcPtnUnboundNativeIrOp.__call__ should "
               "be a list."};
    adt::List<SrcPtnValidIrValue<ValueT, NodeT>> inputs;
    inputs->reserve(input_vals->size());
    for (const auto& input_val : *input_vals) {
      ADT_LET_CONST_REF(input, CastToSrcPtnValidIrValue(input_val));
      inputs->emplace_back(input);
    }
    ADT_LET_CONST_REF(output_vals,
                      axpr::TryGetImpl<adt::List<ValueT>>(args.at(1)))
        << adt::errors::TypeError{
               std::string() +
               "the second argument of SrcPtnUnboundNativeIrOp.__call__ should "
               "be a list."};
    adt::List<UnboundIrValue<ValueT, NodeT>> outputs;
    outputs->reserve(output_vals->size());
    for (const auto& output_val : *output_vals) {
      ADT_LET_CONST_REF(
          output, axpr::TryGetImpl<UnboundIrValue<ValueT, NodeT>>(output_val));
      outputs->emplace_back(output);
    }
    ADT_RETURN_IF_ERR(CheckNoRedundentTensorNames(inputs, outputs));
    ADT_LET_CONST_REF(native_inputs, ConvertInputs(inputs));
    ADT_LET_CONST_REF(native_outputs, ConvertOutputs(outputs));
    ADT_LET_CONST_REF(native_op,
                      Helper{}.GetNativeIrOpByUnboundNativeIrOp(self.value()));
    Helper{}.ConnectIrOpAndIrValue(native_op, native_inputs, native_outputs);
    return adt::Nothing{};
  }

  adt::Result<adt::List<NativeIrValue<NodeT>>> ConvertInputs(
      const adt::List<SrcPtnValidIrValue<ValueT, NodeT>>& inputs) {
    adt::List<NativeIrValue<NodeT>> ret_inputs;
    ret_inputs->reserve(inputs->size());
    using Native = NativeIrValue<NodeT>;
    for (const auto& input : *inputs) {
      const auto& opt_ret_input = input.Match(
          [&](const NativeIrValue<NodeT>& ir_value) -> adt::Result<Native> {
            return ir_value;
          },
          [&](const UnboundIrValue<ValueT, NodeT>& ir_value)
              -> adt::Result<Native> {
            return Helper{}.GetNativeIrValueByUnboundIrValue(ir_value);
          });
      ADT_LET_CONST_REF(ret_input, opt_ret_input);
      ret_inputs->emplace_back(ret_input);
    }
    return ret_inputs;
  }

  adt::Result<adt::List<NativeIrValue<NodeT>>> ConvertOutputs(
      const adt::List<UnboundIrValue<ValueT, NodeT>>& outputs) {
    adt::List<NativeIrValue<NodeT>> ret_outputs;
    ret_outputs->reserve(outputs->size());
    for (const auto& output : *outputs) {
      ADT_LET_CONST_REF(ret_output,
                        Helper{}.GetNativeIrValueByUnboundIrValue(output));
      ret_outputs->emplace_back(ret_output);
    }
    return ret_outputs;
  }

  adt::Result<adt::Ok> CheckNoRedundentTensorNames(
      const adt::List<SrcPtnValidIrValue<ValueT, NodeT>>& inputs,
      const adt::List<UnboundIrValue<ValueT, NodeT>>& outputs) {
    std::unordered_set<std::string> existed_names;
    for (const auto& input : *inputs) {
      existed_names.insert(input.name());
    }
    for (const auto& output : *outputs) {
      ADT_CHECK(existed_names.emplace(output->name).second)
          << adt::errors::TypeError{std::string() + "redundant tensor name '" +
                                    output->name + "' detected."};
    }
    return adt::Ok{};
  }

  adt::Result<SrcPtnValidIrValue<ValueT, NodeT>> CastToSrcPtnValidIrValue(
      const ValueT& arg) {
    return arg.Match(
        [&](const tSrcPtn<NativeIrValue<NodeT>>& value)
            -> adt::Result<SrcPtnValidIrValue<ValueT, NodeT>> {
          return value.value();
        },
        [&](const UnboundIrValue<ValueT, NodeT>& value)
            -> adt::Result<SrcPtnValidIrValue<ValueT, NodeT>> { return value; },
        [&](const auto&) -> adt::Result<SrcPtnValidIrValue<ValueT, NodeT>> {
          return adt::errors::TypeError{
              std::string() +
              "unsupported operand types for "
              "SrcPtnUnboundNativeIrOp.__call__: " +
              axpr::GetTypeName(arg) +
              ". only 'SrcPtnNativeIrValue' and 'UnboundIrValue' supported. "};
        });
  }
};

}  // namespace ap::drr

namespace ap::axpr {

template <typename ValueT, typename NodeT>
struct MethodClassImpl<ValueT,
                       drr::tSrcPtn<drr::UnboundNativeIrOp<ValueT, NodeT>>>
    : public drr::SrcPtnUnboundNativeIrOp<ValueT, NodeT> {};

template <typename ValueT, typename NodeT>
struct MethodClassImpl<
    ValueT,
    TypeImpl<drr::tSrcPtn<drr::UnboundNativeIrOp<ValueT, NodeT>>>> {};

}  // namespace ap::axpr
