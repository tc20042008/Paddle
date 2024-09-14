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
#include "ap/drr/packed_ir_value.h"
#include "ap/drr/tags.h"
#include "ap/drr/unbound_ir_value.h"
#include "ap/drr/unbound_packed_ir_op.h"

namespace ap::drr {

template <typename ValueT, typename NodeT>
struct ResPtnUnboundPackedIrOp {
  using This = ResPtnUnboundPackedIrOp;
  using Self = tResPtn<UnboundPackedIrOp<ValueT, NodeT>>;

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
        "ResPtnUnboundPackedIrOp.__call__ takes 2 arguments. but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(input_vals,
                      axpr::TryGetImpl<adt::List<ValueT>>(args.at(0)))
        << adt::errors::TypeError{
               std::string() +
               "the first argument of ResPtnUnboundPackedIrOp.__call__ should "
               "be a list."};
    adt::List<IrValue<NodeT>> inputs;
    inputs->reserve(input_vals->size());
    for (const auto& input_val : *input_vals) {
      ADT_LET_CONST_REF(input, CastToIrValue(input_val));
      inputs->emplace_back(input);
    }
    ADT_LET_CONST_REF(output_vals,
                      axpr::TryGetImpl<adt::List<ValueT>>(args.at(1)))
        << adt::errors::TypeError{
               std::string() +
               "the second argument of ResPtnUnboundPackedIrOp.__call__ should "
               "be a list."};
    adt::List<IrValue<NodeT>> outputs;
    outputs->reserve(output_vals->size());
    for (const auto& output_val : *output_vals) {
      ADT_LET_CONST_REF(output, CastToIrValue(output_val));
      outputs->emplace_back(output);
    }
    ADT_RETURN_IF_ERROR(CheckNoRedundentTensorNames(inputs, outputs));
    ADT_LET_CONST_REF(packed_op,
                      Helper{}.GetPackedIrOpByUnboundPackedIrOp(self.value()));
    Helper{}.ConnectIrOpAndIrValue(packed_op, inputs, outputs);
    return adt::Nothing{};
  }

  adt::Result<adt::Ok> CheckNoRedundentTensorNames(
      const adt::List<IrValue<NodeT>>& inputs,
      const adt::List<IrValue<NodeT>>& outputs) {
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

  adt::Result<IrValue<NodeT>> CastToIrValue(const ValueT& arg) {
    return arg.Match(
        [&](const tResPtn<NativeIrValue<NodeT>>& value)
            -> adt::Result<IrValue<NodeT>> { return value.value(); },
        [&](const tResPtn<PackedIrValue<NodeT>>& value)
            -> adt::Result<IrValue<NodeT>> { return value.value(); },
        [&](const auto&) -> adt::Result<IrValue<NodeT>> {
          return adt::errors::TypeError{std::string() +
                                        "unsupported operand types for "
                                        "ResPtnUnboundPackedIrOp.__call__: " +
                                        axpr::GetTypeName(arg) +
                                        ". only 'ResPtnNativeIrValue' and "
                                        "'ResPtnPackedIrValue' supported. "};
        });
  }
};

}  // namespace ap::drr

namespace ap::axpr {

template <typename ValueT, typename NodeT>
struct MethodClassImpl<ValueT,
                       drr::tResPtn<drr::UnboundPackedIrOp<ValueT, NodeT>>>
    : public drr::ResPtnUnboundPackedIrOp<ValueT, NodeT> {};

template <typename ValueT, typename NodeT>
struct MethodClassImpl<
    ValueT,
    TypeImpl<drr::tResPtn<drr::UnboundPackedIrOp<ValueT, NodeT>>>> {};

}  // namespace ap::axpr
