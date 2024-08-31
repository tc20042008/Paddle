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

#include "paddle/cinn/adt/adt.h"
#include "paddle/pir/include/core/op_operand.h"
#include "paddle/pir/include/core/op_result.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/dialect/pexpr/core_expr.h"
#include "paddle/pir/include/dialect/pexpr/index_closure.h"
#include "paddle/pir/include/dialect/pexpr/index_expr.h"
#include "paddle/pir/include/dialect/pexpr/op_index_tuple_expr_signature.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

namespace ap {

namespace adt = ::cinn::adt;

}

namespace pir {

class ShapeConstraintIRAnalysis;

}

namespace cinn::dialect {

class FusionOp;

}

namespace ap {

using OpInArg = pir::OpOperand;

// pir::OpResult is more accurate, but there is no pir::Operation method
// returning pir::OpResult object.

struct OpOutArg {
  pir::Value value;
  size_t index;

  bool operator==(const OpOutArg& other) const {
    return other.value == this->value && other.index == this->index;
  }

  std::size_t GetHashValue() const {
    return adt::hash_combine(std::hash<pir::Value>()(value), index);
  }

  static std::optional<OpOutArg> MakeFromValue(pir::Value);
};

}  // namespace ap

namespace std {

template <>
struct hash<ap::OpOutArg> {
  std::size_t operator()(const ap::OpOutArg& out_arg) const {
    return out_arg.GetHashValue();
  }
};

}  // namespace std

namespace ap {

using OpArgImpl = std::variant<OpInArg, OpOutArg>;

struct OpArg : public OpArgImpl {
  using OpArgImpl::OpArgImpl;
  DEFINE_ADT_VARIANT_METHODS(OpArgImpl);

  size_t GetHashValue() const {
    size_t hash_value = Match([&](const auto& impl) {
      return std::hash<std::decay_t<decltype(impl)>>()(impl);
    });
  }
};

}  // namespace ap

namespace std {

template <>
struct hash<ap::OpArg> {
  size_t operator()(const ap::OpArg& op_arg) const {
    return op_arg.GetHashValue();
  }
};

}  // namespace std

namespace ap {

template <typename T>
struct OpArgToImpl {
  std::unordered_map<ap::OpArg, const T> data;

  bool operator==(const OpArgToImpl& other) const {
    return other.data == this->data;
  }
};

template <typename T>
DEFINE_ADT_RC(OpArgTo, OpArgToImpl<T>);

using OpOrArgImpl = std::variant<OpInArg, OpOutArg, pir::Operation*>;

struct OpOrArg : public OpOrArgImpl {
  using OpOrArgImpl::OpOrArgImpl;
  DEFINE_ADT_VARIANT_METHODS(OpOrArgImpl);

  size_t GetHashValue() const {
    return Match([](const auto& impl) {
      return std::hash<std::decay_t<decltype(impl)>>()(impl);
    });
  }
};

struct OpArg2OpIndexesExprSignature {
  OpArgTo<pexpr::OpIndexTupleExprSignature> in_arg2signature;
  OpArgTo<pexpr::OpIndexTupleExprSignature> out_arg2signature;
};

struct Op2Anchor2IndexesExprSignatureImpl {
  std::unordered_map<pir::Operation*, OpArg2OpIndexesExprSignature>
      op2sigatures;

  bool operator==(const Op2Anchor2IndexesExprSignatureImpl& other) const {
    return &other == this;
  }

  bool IsReachable(const OpOrArg& src, const OpOrArg& dst) const;
};
DEFINE_ADT_RC(Op2Anchor2IndexesExprSignature,
              Op2Anchor2IndexesExprSignatureImpl);

struct TrivialFusionDescriptorImpl {
  Op2Anchor2IndexesExprSignature op2anchor2indexes_expr_signature;
  // the input indexes_expr of YieldOp is loop indexes_expr.
  OpArgTo<pexpr::index_expr::RecordableIndexClosure>
      yield_op_arg2custom_index_lambda;

  bool operator==(const TrivialFusionDescriptorImpl& other) const {
    return other.yield_op_arg2custom_index_lambda ==
           this->yield_op_arg2custom_index_lambda;
  }
};
DEFINE_ADT_RC(TrivialFusionDescriptor, const TrivialFusionDescriptorImpl);

using FusionDescriptorImpl = std::variant<TrivialFusionDescriptor>;

struct FusionDescriptor : public FusionDescriptorImpl {
  using FusionDescriptorImpl::FusionDescriptorImpl;
  DEFINE_ADT_VARIANT_METHODS(FusionDescriptorImpl);
};

adt::Result<FusionDescriptor> GetFusionDescriptor(
    const cinn::dialect::FusionOp& op,
    pir::ShapeConstraintIRAnalysis* shape_analysis);

}  // namespace ap
