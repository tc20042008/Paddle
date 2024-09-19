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
#include <vector>
#include "ap/adt/adt.h"
#include "llvm/ADT/SmallVector.h"

namespace ap::graph {

struct NopeVirtualRoot : public std::monostate {
  using std::monostate::monostate;

  std::size_t GetHashValue() const { return 0; }
};

struct NopeNativeValue : public std::monostate {
  using std::monostate::monostate;

  std::size_t GetHashValue() const { return 0; }
};

struct NopePackedValue : public std::monostate {
  using std::monostate::monostate;

  std::size_t GetHashValue() const { return 0; }
};

struct NopeNativeOp {
  std::string op_name;

  bool operator==(const NopeNativeOp& other) const {
    return this->op_name == other.op_name;
  }
  std::size_t GetHashValue() const {
    return std::hash<std::string>()(this->op_name);
  }
};

struct NopePackedOp {
  std::string op_name;

  bool operator==(const NopePackedOp& other) const {
    return this->op_name == other.op_name;
  }
  std::size_t GetHashValue() const {
    return std::hash<std::string>()(this->op_name);
  }
};

struct UpToSink {
  int sink_index;

  bool operator==(const UpToSink& other) const {
    return this->sink_index == other.sink_index;
  }

  std::size_t GetHashValue() const { return this->sink_index; }
};

struct UpToNativeValue : public std::monostate {
  using std::monostate::monostate;
  std::size_t GetHashValue() const { return 0; }
};

struct DownToNativeValue : public std::monostate {
  using std::monostate::monostate;
  std::size_t GetHashValue() const { return 0; }
};

struct UpToPackedValue : public std::monostate {
  using std::monostate::monostate;
  std::size_t GetHashValue() const { return 0; }
};

struct DownToPackedValue : public std::monostate {
  using std::monostate::monostate;
  std::size_t GetHashValue() const { return 0; }
};

struct UpToNativeOpResult {
  int result_index;

  bool operator==(const UpToNativeOpResult& other) const {
    return this->result_index == other.result_index;
  }
  std::size_t GetHashValue() const { return this->result_index; }
};

struct DownToNativeOpResult {
  int result_index;

  bool operator==(const DownToNativeOpResult& other) const {
    return this->result_index == other.result_index;
  }
  std::size_t GetHashValue() const { return this->result_index; }
};

struct UpToPackedOpResult : public std::monostate {
  using std::monostate::monostate;
  std::size_t GetHashValue() const { return 0; }
};

struct DownToPackedOpResult : public std::monostate {
  using std::monostate::monostate;
  std::size_t GetHashValue() const { return 0; }
};

struct UpToNativeOp {
  std::string op_name;

  bool operator==(const UpToNativeOp& other) const {
    return this->op_name == other.op_name;
  }
  std::size_t GetHashValue() const {
    return std::hash<std::string>()(this->op_name);
  }
};

struct DownToNativeOp {
  std::string op_name;

  bool operator==(const DownToNativeOp& other) const {
    return this->op_name == other.op_name;
  }
  std::size_t GetHashValue() const {
    return std::hash<std::string>()(this->op_name);
  }
};

struct UpToPackedOp {
  std::string op_name;

  bool operator==(const UpToPackedOp& other) const {
    return this->op_name == other.op_name;
  }
  std::size_t GetHashValue() const {
    return std::hash<std::string>()(this->op_name);
  }
};

struct DownToPackedOp {
  std::string op_name;

  bool operator==(const DownToPackedOp& other) const {
    return this->op_name == other.op_name;
  }
  std::size_t GetHashValue() const {
    return std::hash<std::string>()(this->op_name);
  }
};

struct UpToNativeOpOperand {
  int operand_index;

  bool operator==(const UpToNativeOpOperand& other) const {
    return this->operand_index == other.operand_index;
  }
  std::size_t GetHashValue() const { return this->operand_index; }
};

struct DownToNativeOpOperand {
  int operand_index;

  bool operator==(const DownToNativeOpOperand& other) const {
    return this->operand_index == other.operand_index;
  }
  std::size_t GetHashValue() const { return this->operand_index; }
};

struct UpToPackedOpOperand : public std::monostate {
  using std::monostate::monostate;
  std::size_t GetHashValue() const { return 0; }
};

struct DownToPackedOpOperand : public std::monostate {
  using std::monostate::monostate;
  std::size_t GetHashValue() const { return 0; }
};

// NopeXXX means no jump, usually used for initial node.

using FreeJumpImpl = std::variant<NopeNativeValue,
                                  NopePackedValue,
                                  NopeNativeOp,
                                  NopePackedOp,
                                  UpToNativeValue,
                                  DownToNativeValue,
                                  UpToPackedValue,
                                  DownToPackedValue,
                                  UpToNativeOpResult,
                                  DownToNativeOpResult,
                                  UpToPackedOpResult,
                                  DownToPackedOpResult,
                                  UpToNativeOp,
                                  DownToNativeOp,
                                  UpToPackedOp,
                                  DownToPackedOp,
                                  UpToNativeOpOperand,
                                  DownToNativeOpOperand,
                                  UpToPackedOpOperand,
                                  DownToPackedOpOperand>;

struct FreeJump : public FreeJumpImpl {
  using FreeJumpImpl::FreeJumpImpl;
  DEFINE_ADT_VARIANT_METHODS(FreeJumpImpl);

  std::size_t GetHashValue() const {
    std::size_t hash_value =
        Match([](const auto& impl) { return impl.GetHashValue(); });
    return adt::hash_combine(hash_value, this->index());
  }
};

using UpJumpImpl = std::variant<NopeNativeValue,
                                NopePackedValue,
                                NopeNativeOp,
                                NopePackedOp,
                                UpToNativeValue,
                                UpToPackedValue,
                                UpToNativeOpResult,
                                UpToPackedOpResult,
                                UpToNativeOp,
                                UpToPackedOp,
                                UpToNativeOpOperand,
                                UpToPackedOpOperand>;

struct UpJump : public UpJumpImpl {
  using UpJumpImpl::UpJumpImpl;
  DEFINE_ADT_VARIANT_METHODS(UpJumpImpl);

  std::size_t GetHashValue() const {
    std::size_t hash_value =
        Match([](const auto& impl) { return impl.GetHashValue(); });
    return adt::hash_combine(hash_value, this->index());
  }
};

using DownJumpImpl = std::variant<NopeNativeValue,
                                  NopePackedValue,
                                  NopeNativeOp,
                                  NopePackedOp,
                                  DownToNativeValue,
                                  DownToPackedValue,
                                  DownToNativeOpResult,
                                  DownToPackedOpResult,
                                  DownToNativeOp,
                                  DownToPackedOp,
                                  DownToNativeOpOperand,
                                  DownToPackedOpOperand>;

struct DownJump : public DownJumpImpl {
  using DownJumpImpl::DownJumpImpl;
  DEFINE_ADT_VARIANT_METHODS(DownJumpImpl);

  std::size_t GetHashValue() const {
    std::size_t hash_value =
        Match([](const auto& impl) { return impl.GetHashValue(); });
    return adt::hash_combine(hash_value, this->index());
  }
};

using SinkUpToOtherJumpImpl = std::variant<NopeVirtualRoot,
                                           UpToSink,
                                           UpToNativeValue,
                                           UpToPackedValue,
                                           UpToNativeOpResult,
                                           UpToPackedOpResult,
                                           UpToNativeOp,
                                           UpToPackedOp,
                                           UpToNativeOpOperand,
                                           UpToPackedOpOperand>;

struct SinkUpToOtherJump : public SinkUpToOtherJumpImpl {
  using SinkUpToOtherJumpImpl::SinkUpToOtherJumpImpl;
  DEFINE_ADT_VARIANT_METHODS(SinkUpToOtherJumpImpl);

  std::size_t GetHashValue() const {
    std::size_t hash_value =
        Match([](const auto& impl) { return impl.GetHashValue(); });
    return adt::hash_combine(hash_value, this->index());
  }
};

}  // namespace ap::graph

namespace std {

template <>
struct hash<::ap::graph::FreeJump> {
  std::size_t operator()(const ::ap::graph::FreeJump& jump) const {
    return jump.GetHashValue();
  }
};

template <>
struct hash<std::vector<::ap::graph::FreeJump>> {
  std::size_t operator()(
      const std::vector<::ap::graph::FreeJump>& jumps) const {
    std::size_t hash_value = std::hash<const char*>()("FreeJump");
    for (const auto& jump : jumps) {
      hash_value = ::cinn::adt::hash_combine(hash_value, jump.GetHashValue());
    }
    return hash_value;
  }
};

template <>
struct hash<::ap::graph::UpJump> {
  std::size_t operator()(const ::ap::graph::UpJump& jump) const {
    return jump.GetHashValue();
  }
};

template <int Num>
struct hash<llvm::SmallVector<::ap::graph::UpJump, Num>> {
  std::size_t operator()(
      const llvm::SmallVector<::ap::graph::UpJump, Num>& jumps) const {
    std::size_t hash_value = std::hash<const char*>()("UpJump");
    for (const auto& jump : jumps) {
      hash_value = ::cinn::adt::hash_combine(hash_value, jump.GetHashValue());
    }
    return hash_value;
  }
};

template <>
struct hash<::ap::graph::DownJump> {
  std::size_t operator()(const ::ap::graph::DownJump& jump) const {
    return jump.GetHashValue();
  }
};

template <int Num>
struct hash<llvm::SmallVector<::ap::graph::DownJump, Num>> {
  std::size_t operator()(
      const llvm::SmallVector<::ap::graph::DownJump, Num>& jumps) const {
    std::size_t hash_value = std::hash<const char*>()("DownJump");
    for (const auto& jump : jumps) {
      hash_value = ::cinn::adt::hash_combine(hash_value, jump.GetHashValue());
    }
    return hash_value;
  }
};

template <>
struct hash<::ap::graph::SinkUpToOtherJump> {
  std::size_t operator()(const ::ap::graph::SinkUpToOtherJump& jump) const {
    return jump.GetHashValue();
  }
};

template <>
struct hash<std::vector<::ap::graph::SinkUpToOtherJump>> {
  std::size_t operator()(
      const std::vector<::ap::graph::SinkUpToOtherJump>& jumps) const {
    std::size_t hash_value = std::hash<const char*>()("SinkUpToOtherJump");
    for (const auto& jump : jumps) {
      hash_value = ::cinn::adt::hash_combine(hash_value, jump.GetHashValue());
    }
    return hash_value;
  }
};

}  // namespace std
