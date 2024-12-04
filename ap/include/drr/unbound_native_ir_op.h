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

#include "ap/include/adt/adt.h"
#include "ap/include/axpr/type.h"
#include "ap/include/axpr/value.h"
#include "ap/include/drr/native_ir_op_declare.h"
#include "ap/include/drr/tags.h"
#include "ap/include/drr/type.h"
#include "ap/include/graph/node.h"

namespace ap::drr {

template <typename NodeT>
struct UnboundNativeIrOpImpl {
  NativeIrOpDeclare<NodeT> op_declare;
  std::string name;

  bool operator==(const UnboundNativeIrOpImpl& other) const {
    return this->op_declare == other.op_declare && this->name == other.name;
  }
};

template <typename NodeT>
DEFINE_ADT_RC(UnboundNativeIrOp, UnboundNativeIrOpImpl<NodeT>);

const axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>&
GetSrcPtnUnboundNativeIrOpClass();

template <typename NodeT>
struct Type<drr::tSrcPtn<drr::UnboundNativeIrOp<NodeT>>>
    : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "SrcPtnUnboundNativeIrOp"; }

  static const axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>&
  GetClass() {
    return GetSrcPtnUnboundNativeIrOpClass();
  }
};

const axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>&
GetResPtnUnboundNativeIrOpClass();

template <typename NodeT>
struct Type<drr::tResPtn<drr::UnboundNativeIrOp<NodeT>>>
    : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "ResPtnUnboundNativeIrOp"; }

  static const axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>&
  GetClass() {
    return GetResPtnUnboundNativeIrOpClass();
  }
};

}  // namespace ap::drr
