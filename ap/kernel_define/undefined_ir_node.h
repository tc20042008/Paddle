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

#include <functional>
#include "ap/adt/adt.h"
#include "ap/axpr/type.h"

namespace ap::kernel_define {

struct UndefinedDimExpr : public std::monostate {
  using std::monostate::monostate;
};

struct UndefinedNativeIrOp : public std::monostate {
  using std::monostate::monostate;
};

struct UndefinedPackedIrOp : public std::monostate {
  using std::monostate::monostate;
};

struct UndefinedNativeIrValue : public std::monostate {
  using std::monostate::monostate;
};

struct UndefinedPackedIrValue : public std::monostate {
  using std::monostate::monostate;
};

using UndefinedIrNodeImpl = std::variant<UndefinedNativeIrOp,
                                         UndefinedPackedIrOp,
                                         UndefinedNativeIrValue,
                                         UndefinedPackedIrValue>;

struct UndefinedIrNode : public UndefinedIrNodeImpl {
  using UndefinedIrNodeImpl::UndefinedIrNodeImpl;
  using dim_expr_type = UndefinedDimExpr;
  using native_op_type = UndefinedNativeIrOp;
  using packed_op_type = UndefinedPackedIrOp;
  using native_value_type = UndefinedNativeIrValue;
  using packed_value_type = UndefinedPackedIrValue;
  DEFINE_ADT_VARIANT_METHODS(UndefinedIrNodeImpl);

  std::size_t GetHashValue() const { return this->index(); }
};

}  // namespace ap::kernel_define

namespace std {

template <>
struct hash<ap::kernel_define::UndefinedIrNode> {
  std::size_t operator()(const ap::kernel_define::UndefinedIrNode& node) const {
    return node.GetHashValue();
  }
};

}  // namespace std

namespace ap::axpr {

template <>
struct TypeImpl<kernel_define::UndefinedDimExpr> : public std::monostate {
  using std::monostate::monostate;

  const char* Name() const { return "UndefinedDimExpr"; }
};

template <>
struct TypeImpl<kernel_define::UndefinedNativeIrValue> : public std::monostate {
  using std::monostate::monostate;

  const char* Name() const { return "UndefinedNativeIrValue"; }
};

template <>
struct TypeImpl<kernel_define::UndefinedPackedIrValue> : public std::monostate {
  using std::monostate::monostate;

  const char* Name() const { return "UndefinedPackedIrValue"; }
};

template <>
struct TypeImpl<kernel_define::UndefinedNativeIrOp> : public std::monostate {
  using std::monostate::monostate;

  const char* Name() const { return "UndefinedNativeIrOp"; }
};

template <>
struct TypeImpl<kernel_define::UndefinedPackedIrOp> : public std::monostate {
  using std::monostate::monostate;

  const char* Name() const { return "UndefinedPackedIrOp"; }
};

}  // namespace ap::axpr
