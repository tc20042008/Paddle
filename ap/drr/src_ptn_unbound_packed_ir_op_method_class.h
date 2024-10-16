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
#include "ap/drr/op_tensor_pattern_ctx_helper.h"
#include "ap/drr/packed_ir_value.h"
#include "ap/drr/tags.h"
#include "ap/drr/unbound_ir_value.h"
#include "ap/drr/unbound_packed_ir_op.h"

namespace ap::drr {

template <typename ValueT, typename NodeT>
using SrcPtnValidInIrValueImpl =
    std::variant<PackedIrValue<NodeT>,
                 NativeIrValue<NodeT>,
                 UnboundIrValue<ValueT, NodeT>,
                 UnboundPackedIrValue<ValueT, NodeT>>;

template <typename ValueT, typename NodeT>
struct SrcPtnValidInIrValue : public SrcPtnValidInIrValueImpl<ValueT, NodeT> {
  using SrcPtnValidInIrValueImpl<ValueT, NodeT>::SrcPtnValidInIrValueImpl;

  DEFINE_ADT_VARIANT_METHODS(SrcPtnValidInIrValueImpl<ValueT, NodeT>);

  const std::string& name() const {
    return Match([](const auto& ir_value) -> const std::string& {
      return ir_value->name;
    });
  }
};

template <typename ValueT, typename NodeT>
using SrcPtnValidOutIrValueImpl =
    std::variant<UnboundIrValue<ValueT, NodeT>,
                 UnboundPackedIrValue<ValueT, NodeT>>;

template <typename ValueT, typename NodeT>
struct SrcPtnValidOutIrValue : public SrcPtnValidOutIrValueImpl<ValueT, NodeT> {
  using SrcPtnValidOutIrValueImpl<ValueT, NodeT>::SrcPtnValidOutIrValueImpl;

  DEFINE_ADT_VARIANT_METHODS(SrcPtnValidOutIrValueImpl<ValueT, NodeT>);

  const std::string& name() const {
    return Match([](const auto& ir_value) -> const std::string& {
      return ir_value->name;
    });
  }
};

template <typename ValueT, typename NodeT>
struct SrcPtnUnboundPackedIrOp {
  using This = SrcPtnUnboundPackedIrOp;
  using Self = tSrcPtn<UnboundPackedIrOp<ValueT, NodeT>>;

  adt::Result<ValueT> ToString(const Self& self) {
    std::ostringstream ss;
    const void* ptr = self.value().__adt_rc_shared_ptr_raw_ptr();
    ss << "<" << axpr::TypeImpl<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

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
        "SrcPtnUnboundPackedIrOp.__call__ takes 2 arguments. but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(input_vals,
                      axpr::TryGetImpl<adt::List<ValueT>>(args.at(0)))
        << adt::errors::TypeError{
               std::string() +
               "the first argument of SrcPtnUnboundPackedIrOp.__call__ should "
               "be a list."};
    adt::List<SrcPtnValidInIrValue<ValueT, NodeT>> inputs;
    inputs->reserve(input_vals->size());
    for (const auto& input_val : *input_vals) {
      ADT_LET_CONST_REF(input, CastToSrcPtnValidInIrValue(input_val));
      inputs->emplace_back(input);
    }
    ADT_LET_CONST_REF(output_vals,
                      axpr::TryGetImpl<adt::List<ValueT>>(args.at(1)))
        << adt::errors::TypeError{
               std::string() +
               "the second argument of SrcPtnUnboundPackedIrOp.__call__ should "
               "be a list."};
    adt::List<SrcPtnValidOutIrValue<ValueT, NodeT>> outputs;
    outputs->reserve(output_vals->size());
    for (const auto& output_val : *output_vals) {
      ADT_LET_CONST_REF(output, CastToSrcPtnValidOutIrValue(output_val));
      outputs->emplace_back(output);
    }
    ADT_RETURN_IF_ERR(CheckNoRedundentTensorNames(inputs, outputs));
    ADT_LET_CONST_REF(opt_packed_inputs, ConvertInputs(inputs));
    ADT_LET_CONST_REF(opt_packed_outputs, ConvertOutputs(outputs));
    ADT_LET_CONST_REF(packed_op,
                      Helper{}.GetPackedIrOpByUnboundPackedIrOp(self.value()));
    Helper{}.ConnectIrOpAndIrValue(
        packed_op, opt_packed_inputs, opt_packed_outputs);
    return adt::Nothing{};
  }

  adt::Result<adt::List<IrValue<NodeT>>> ConvertInputs(
      const adt::List<SrcPtnValidInIrValue<ValueT, NodeT>>& inputs) {
    adt::List<IrValue<NodeT>> ret_inputs;
    ret_inputs->reserve(inputs->size());
    using IrVal = IrValue<NodeT>;
    for (const auto& input : *inputs) {
      const auto& opt_ret_input = input.Match(
          [&](const NativeIrValue<NodeT>& ir_value) -> adt::Result<IrVal> {
            return ir_value;
          },
          [&](const PackedIrValue<NodeT>& ir_value) -> adt::Result<IrVal> {
            return ir_value;
          },
          [&](const UnboundIrValue<ValueT, NodeT>& ir_value)
              -> adt::Result<IrVal> {
            ADT_LET_CONST_REF(
                ret, Helper{}.GetNativeIrValueByUnboundIrValue(ir_value));
            return ret;
          },
          [&](const UnboundPackedIrValue<ValueT, NodeT>& ir_value)
              -> adt::Result<IrVal> {
            ADT_LET_CONST_REF(
                ret, Helper{}.GetPackedIrValueByUnboundPackedIrValue(ir_value));
            return ret;
          });
      ADT_LET_CONST_REF(ret_input, opt_ret_input);
      ret_inputs->emplace_back(ret_input);
    }
    return ret_inputs;
  }

  adt::Result<adt::List<IrValue<NodeT>>> ConvertOutputs(
      const adt::List<SrcPtnValidOutIrValue<ValueT, NodeT>>& outputs) {
    adt::List<IrValue<NodeT>> ret_outputs;
    using IrVal = IrValue<NodeT>;
    ret_outputs->reserve(outputs->size());
    for (const auto& output : *outputs) {
      const auto& opt_ret_output = output.Match(
          [&](const UnboundIrValue<ValueT, NodeT>& ir_value)
              -> adt::Result<IrVal> {
            ADT_LET_CONST_REF(
                ret, Helper{}.GetNativeIrValueByUnboundIrValue(ir_value));
            return ret;
          },
          [&](const UnboundPackedIrValue<ValueT, NodeT>& ir_value)
              -> adt::Result<IrVal> {
            ADT_LET_CONST_REF(
                ret, Helper{}.GetPackedIrValueByUnboundPackedIrValue(ir_value));
            return ret;
          });
      ADT_LET_CONST_REF(ret_output, opt_ret_output);
      ret_outputs->emplace_back(ret_output);
    }
    return ret_outputs;
  }

  adt::Result<adt::Ok> CheckNoRedundentTensorNames(
      const adt::List<SrcPtnValidInIrValue<ValueT, NodeT>>& inputs,
      const adt::List<SrcPtnValidOutIrValue<ValueT, NodeT>>& outputs) {
    std::unordered_set<std::string> existed_names;
    for (const auto& input : *inputs) {
      existed_names.insert(input.name());
    }
    for (const auto& output : *outputs) {
      ADT_CHECK(existed_names.emplace(output.name()).second)
          << adt::errors::TypeError{std::string() + "redundant tensor name '" +
                                    output.name() + "' detected."};
    }
    return adt::Ok{};
  }

  adt::Result<SrcPtnValidInIrValue<ValueT, NodeT>> CastToSrcPtnValidInIrValue(
      const ValueT& arg) {
    return arg.Match(
        [&](const tSrcPtn<NativeIrValue<NodeT>>& value)
            -> adt::Result<SrcPtnValidInIrValue<ValueT, NodeT>> {
          return SrcPtnValidInIrValue<ValueT, NodeT>{value.value()};
        },
        [&](const tSrcPtn<PackedIrValue<NodeT>>& value)
            -> adt::Result<SrcPtnValidInIrValue<ValueT, NodeT>> {
          return SrcPtnValidInIrValue<ValueT, NodeT>{value.value()};
        },
        [&](const UnboundIrValue<ValueT, NodeT>& value)
            -> adt::Result<SrcPtnValidInIrValue<ValueT, NodeT>> {
          return value;
        },
        [&](const UnboundPackedIrValue<ValueT, NodeT>& value)
            -> adt::Result<SrcPtnValidInIrValue<ValueT, NodeT>> {
          return value;
        },
        [&](const auto&) -> adt::Result<SrcPtnValidInIrValue<ValueT, NodeT>> {
          return adt::errors::TypeError{
              std::string() +
              "unsupported operand types for the first arugments of "
              "SrcPtnUnboundPackedIrOp.__call__: " +
              axpr::GetTypeName(arg) +
              ". only 'SrcPtnPackedIrValue' and 'UnboundIrValue' supported. "};
        });
  }

  adt::Result<SrcPtnValidOutIrValue<ValueT, NodeT>> CastToSrcPtnValidOutIrValue(
      const ValueT& arg) {
    return arg.Match(
        [&](const UnboundIrValue<ValueT, NodeT>& value)
            -> adt::Result<SrcPtnValidOutIrValue<ValueT, NodeT>> {
          return value;
        },
        [&](const UnboundPackedIrValue<ValueT, NodeT>& value)
            -> adt::Result<SrcPtnValidOutIrValue<ValueT, NodeT>> {
          return value;
        },
        [&](const auto&) -> adt::Result<SrcPtnValidOutIrValue<ValueT, NodeT>> {
          return adt::errors::TypeError{
              std::string() +
              "unsupported operand types for the second arguments of "
              "SrcPtnUnboundPackedIrOp.__call__: " +
              axpr::GetTypeName(arg) +
              ". only 'SrcPtnPackedIrValue' and 'UnboundIrValue' supported. "};
        });
  }
};

}  // namespace ap::drr

namespace ap::axpr {

template <typename ValueT, typename NodeT>
struct MethodClassImpl<ValueT,
                       drr::tSrcPtn<drr::UnboundPackedIrOp<ValueT, NodeT>>>
    : public drr::SrcPtnUnboundPackedIrOp<ValueT, NodeT> {};

template <typename ValueT, typename NodeT>
struct MethodClassImpl<
    ValueT,
    TypeImpl<drr::tSrcPtn<drr::UnboundPackedIrOp<ValueT, NodeT>>>> {};

}  // namespace ap::axpr
