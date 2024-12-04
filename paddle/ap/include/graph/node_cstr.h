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

#include "paddle/ap/include/graph/adt.h"

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

struct OptPackedIrOpCstr {
  PackedIrOpCstr packed_ir_op_cstr;

  bool operator==(const OptPackedIrOpCstr& other) const {
    return this->packed_ir_op_cstr == other.packed_ir_op_cstr;
  }
};

struct OptPackedIrOpOperandCstr {
  PackedIrOpOperandCstr packed_ir_op_operand_cstr;

  bool operator==(const OptPackedIrOpOperandCstr& other) const {
    return this->packed_ir_op_operand_cstr == other.packed_ir_op_operand_cstr;
  }
};

struct OptPackedIrOpResultCstr : public std::monostate {
  PackedIrOpResultCstr packed_ir_op_result_cstr;

  bool operator==(const OptPackedIrOpResultCstr& other) const {
    return this->packed_ir_op_result_cstr == other.packed_ir_op_result_cstr;
  }
};

struct RefIrValueCstr : public std::monostate {
  using std::monostate::monostate;
};

struct RefIrOpCstr : public std::monostate {
  using std::monostate::monostate;
};

struct RefIrOpOperandCstr : public std::monostate {
  using std::monostate::monostate;
};

struct RefIrOpResultCstr : public std::monostate {
  using std::monostate::monostate;
};

using NodeCstrImpl = std::variant<NativeIrValueCstr,
                                  NativeIrOpCstr,
                                  NativeIrOpOperandCstr,
                                  NativeIrOpResultCstr,
                                  PackedIrValueCstr,
                                  PackedIrOpCstr,
                                  PackedIrOpOperandCstr,
                                  PackedIrOpResultCstr,
                                  OptPackedIrOpCstr,
                                  OptPackedIrOpOperandCstr,
                                  OptPackedIrOpResultCstr,
                                  RefIrValueCstr,
                                  RefIrOpCstr,
                                  RefIrOpOperandCstr,
                                  RefIrOpResultCstr>;
// node constraint
struct NodeCstr : public NodeCstrImpl {
  using NodeCstrImpl::NodeCstrImpl;
  DEFINE_ADT_VARIANT_METHODS(NodeCstrImpl);

  adt::Result<bool> Satisfy(const NodeCstr& sg_node_cstr) const {
    using RetT = adt::Result<bool>;
    const auto& pattern_match = ::common::Overloaded{
        [&](const PackedIrOpCstr& bg_cstr, const OptPackedIrOpCstr& sg_cstr)
            -> RetT { return bg_cstr == sg_cstr.packed_ir_op_cstr; },
        [&](const PackedIrOpOperandCstr& bg_cstr,
            const OptPackedIrOpOperandCstr& sg_cstr) -> RetT {
          return bg_cstr == sg_cstr.packed_ir_op_operand_cstr;
        },
        [&](const PackedIrOpResultCstr& bg_cstr,
            const OptPackedIrOpResultCstr& sg_cstr) -> RetT {
          return bg_cstr == sg_cstr.packed_ir_op_result_cstr;
        },
        [&](const RefIrValueCstr& bg_cstr,
            const NativeIrValueCstr& sg_cstr) -> RetT { return true; },
        [&](const RefIrOpCstr& bg_cstr,
            const OptPackedIrOpCstr& sg_cstr) -> RetT { return true; },
        [&](const RefIrOpOperandCstr& bg_cstr,
            const OptPackedIrOpOperandCstr& sg_cstr) -> RetT { return true; },
        [&](const RefIrOpResultCstr& bg_cstr,
            const OptPackedIrOpResultCstr& sg_cstr) -> RetT { return true; },
        [&](const auto&, const auto&) -> RetT {
          return *this == sg_node_cstr;
        }};
    return std::visit(pattern_match, this->variant(), sg_node_cstr.variant());
  }
};

struct SmallGraphNodeCstr {
  NodeCstr node_cstr;
};

struct BigGraphNodeCstr {
  NodeCstr node_cstr;

  adt::Result<bool> Satisfy(const SmallGraphNodeCstr& sg_node_cstr) const {
    return this->node_cstr.Satisfy(sg_node_cstr.node_cstr);
  }
};

}  // namespace ap::graph
