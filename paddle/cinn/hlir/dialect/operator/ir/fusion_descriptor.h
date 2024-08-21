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

#include "paddle/pir/include/dialect/pexpr/adt.h"
#include "paddle/pir/include/dialect/pexpr/core_expr.h"
#include "paddle/pir/include/dialect/pexpr/index_expr.h"
#include "paddle/pir/include/dialect/pexpr/index_lambda.h"

namespace pir {

class Operation;

}

namespace ap {

struct TrivialFusionDescriptorImpl {
  template <typename T>
  using AnchorableOutIdxTo = std::vector<std::optional<T>>;

  using LoopIndexesExprConverters = std::vector<pexpr::IndexLambda>;

  AnchorableOutIdxTo<LoopIndexesExprConverters> loop_indexes_expr_converters;
  std::unordered_map<const pir::Operation*, pexpr::IndexLambda>
      op2custom_index_lambda;

  bool operator==(const TrivialFusionDescriptorImpl& other) const {
    return other.loop_indexes_expr_converters ==
               this->loop_indexes_expr_converters &&
           other.op2custom_index_lambda == this->op2custom_index_lambda;
  }
};
DEFINE_ADT_RC(TrivialFusionDescriptor, const TrivialFusionDescriptorImpl);

using FusionDescriptorImpl = std::variant<TrivialFusionDescriptor>;

struct FusionDescriptor : public FusionDescriptorImpl {
  using FusionDescriptorImpl::FusionDescriptorImpl;
  DEFINE_ADT_VARIANT_METHODS(FusionDescriptorImpl);
};

}  // namespace ap
