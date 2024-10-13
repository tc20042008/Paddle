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

#include "ap/graph/adt.h"

namespace ap::graph {

struct NativeIrValueCstr : public std::monostate {
  using std::monostate::monostate;
};

struct NativeIrOpCstr {
  std::string op_name;

  bool operator==(const NativeIrOpCstr& other) const {
    return this->op_name == other.op_name;
  }
};

struct NativeIrOpOperandCstr {
  std::size_t index;

  bool operator==(const NativeIrOpOperandCstr& other) const {
    return this->index == other.index;
  }
};

struct NativeIrOpResultCstr {
  std::size_t index;

  bool operator==(const NativeIrOpResultCstr& other) const {
    return this->index == other.index;
  }
};

struct PackedIrValueCstr {
  bool operator==(const PackedIrValueCstr&) const { return false; }
  bool operator!=(const PackedIrValueCstr&) const { return false; }
};

struct PackedIrOpCstr {
  std::string op_name;

  bool operator==(const PackedIrOpCstr& other) const {
    return this->op_name == other.op_name;
  }
};

struct PackedIrOpOperandCstr : public std::monostate {
  using std::monostate::monostate;
};

struct PackedIrOpResultCstr : public std::monostate {
  using std::monostate::monostate;
};

using NodeCstrImpl = std::variant<NativeIrValueCstr,
                                  NativeIrOpCstr,
                                  NativeIrOpOperandCstr,
                                  NativeIrOpResultCstr,
                                  PackedIrValueCstr,
                                  PackedIrOpCstr,
                                  PackedIrOpOperandCstr,
                                  PackedIrOpResultCstr>;
// node constraint
struct NodeCstr : public NodeCstrImpl {
  using NodeCstrImpl::NodeCstrImpl;
  DEFINE_ADT_VARIANT_METHODS(NodeCstrImpl);
};

}  // namespace ap::graph
