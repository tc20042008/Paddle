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
#include "paddle/pir/include/dialect/pexpr/binary_func.h"
#include "paddle/pir/include/dialect/pexpr/type.h"
#include "paddle/pir/include/dialect/pexpr/unary_func.h"

namespace pexpr {

inline constexpr const char* kBuiltinIf() { return "if"; }
inline constexpr const char* kBuiltinApply() { return "__builtin_apply__"; }
inline constexpr const char* kBuiltinNothing() { return "None"; }
inline constexpr const char* kBuiltinId() { return "__builtin_identity__"; }
inline constexpr const char* kBuiltinList() { return "__builtin_list__"; }
inline constexpr const char* kBuiltinCall() { return "__builtin_call__"; }
inline constexpr const char* kBuiltinGetAttr() { return "__builtin_getattr__"; }
inline constexpr const char* kBuiltinGetItem() { return "__builtin_getitem__"; }
inline constexpr const char* kBuiltinReturn() { return "__builtin_return__"; }

#define DEFINE_PEXPR_BUILTIN_CONSTANT_NAME(name, op) \
  inline constexpr const char* kBuiltin##name() {    \
    return "__builtin_" #name "__";                  \
  }
PEXPR_FOR_EACH_BINARY_OP(DEFINE_PEXPR_BUILTIN_CONSTANT_NAME)
PEXPR_FOR_EACH_UNARY_OP(DEFINE_PEXPR_BUILTIN_CONSTANT_NAME)
#undef DEFINE_PEXPR_BUILTIN_CONSTANT_NAME

namespace builtin_symbol {

struct If : public std::monostate {
  using std::monostate::monostate;
  static constexpr const char* Name() { return kBuiltinIf(); }
  std::size_t GetHashValue() const { return 0; }
};

struct Apply : public std::monostate {
  using std::monostate::monostate;
  static constexpr const char* Name() { return kBuiltinApply(); }
  std::size_t GetHashValue() const { return 0; }
};

struct Nothing : public std::monostate {
  using std::monostate::monostate;
  static constexpr const char* Name() { return kBuiltinNothing(); }

  std::size_t GetHashValue() const { return 0; }
};

struct Id : public std::monostate {
  using std::monostate::monostate;
  static constexpr const char* Name() { return kBuiltinId(); }
  std::size_t GetHashValue() const { return 0; }
};

struct List : public std::monostate {
  using std::monostate::monostate;
  static constexpr const char* Name() { return kBuiltinList(); }
  std::size_t GetHashValue() const { return 0; }
};

struct Call : public std::monostate {
  using std::monostate::monostate;
  static constexpr const char* Name() { return kBuiltinCall(); }
  static constexpr int num_operands = 1;
  std::size_t GetHashValue() const { return 0; }
};

struct GetAttr : public std::monostate {
  using std::monostate::monostate;
  static constexpr const char* Name() { return kBuiltinGetAttr(); }
  static constexpr int num_operands = 2;
  std::size_t GetHashValue() const { return 0; }
};

struct GetItem : public std::monostate {
  using std::monostate::monostate;
  static constexpr const char* Name() { return kBuiltinGetItem(); }
  static constexpr int num_operands = 2;
  std::size_t GetHashValue() const { return 0; }
};

#define DEFINE_UNARY_SYMBOL(name, op)                                \
  struct name : public std::monostate {                              \
    using std::monostate::monostate;                                 \
    static constexpr const char* Name() { return kBuiltin##name(); } \
    static constexpr int num_operands = 1;                           \
    std::size_t GetHashValue() const { return 0; }                   \
  };

PEXPR_FOR_EACH_UNARY_OP(DEFINE_UNARY_SYMBOL);

#undef DEFINE_UNARY_SYMBOL;

#define DEFINE_BINARY_SYMBOL(name, op)                               \
  struct name : public std::monostate {                              \
    using std::monostate::monostate;                                 \
    static constexpr const char* Name() { return kBuiltin##name(); } \
    static constexpr int num_operands = 2;                           \
    std::size_t GetHashValue() const { return 0; }                   \
  };

PEXPR_FOR_EACH_BINARY_OP(DEFINE_BINARY_SYMBOL);

#undef DEFINE_BINARY_SYMBOL;

using OpImpl = std::variant<
#define MAKE_OP_IMPL_ALTENATIVE(name, op) name,
    PEXPR_FOR_EACH_BINARY_OP(MAKE_OP_IMPL_ALTENATIVE)
        PEXPR_FOR_EACH_UNARY_OP(MAKE_OP_IMPL_ALTENATIVE)
#undef MAKE_OP_IMPL_ALTENATIVE
            Call,
    GetAttr,
    GetItem>;

struct Op : public OpImpl {
  using OpImpl::OpImpl;
  DEFINE_ADT_VARIANT_METHODS(OpImpl);

  const char* Name() const {
    return Match([](const auto& impl) { return impl.Name(); });
  }

  std::size_t GetHashValue() const {
    std::size_t hash_value =
        Match([&](const auto& impl) { return impl.GetHashValue(); });
    return adt::hash_combine(hash_value, this->index());
  }
};

using SymbolImpl = std::variant<If, Apply, Nothing, Id, List, Op>;

struct Symbol : public SymbolImpl {
  using SymbolImpl::SymbolImpl;
  DEFINE_ADT_VARIANT_METHODS(SymbolImpl);

  const char* Name() const {
    return Match([](const auto& impl) { return impl.Name(); });
  }

  std::size_t GetHashValue() const {
    std::size_t hash_value =
        Match([&](const auto& impl) { return impl.GetHashValue(); });
    return adt::hash_combine(hash_value, this->index());
  }
};

inline adt::Maybe<Symbol> GetSymbolFromString(const std::string& name) {
  static const std::unordered_map<std::string, Symbol> map{
      {If::Name(), If{}},
      {Apply::Name(), Apply{}},
      {Nothing::Name(), Nothing{}},
      {Id::Name(), Id{}},
      {List::Name(), List{}},
      {Call::Name(), Op{Call{}}},
      {GetAttr::Name(), Op{GetAttr{}}},
      {GetItem::Name(), Op{GetItem{}}},
#define MAKE_SYMBOL_ENTRY(cls, op) {cls::Name(), Op{cls{}}},
      PEXPR_FOR_EACH_BINARY_OP(MAKE_SYMBOL_ENTRY)
          PEXPR_FOR_EACH_UNARY_OP(MAKE_SYMBOL_ENTRY)
#undef MAKE_SYMBOL_ENTRY
  };
  const auto& iter = map.find(name);
  if (iter == map.end()) {
    return adt::Nothing{};
  }
  return iter->second;
}

}  // namespace builtin_symbol

template <typename BuiltinSymbol>
struct ConvertBuiltinSymbolToArithmetic {
  static const bool convertable = false;
  using arithmetic_op_type = void;
};

#define SPECIALIZE_ConvertBuiltinSymbolToArithmetic(cls, op)     \
  template <>                                                    \
  struct ConvertBuiltinSymbolToArithmetic<builtin_symbol::cls> { \
    static const bool convertable = true;                        \
    using arithmetic_op_type = Arithmetic##cls;                  \
  };

PEXPR_FOR_EACH_BINARY_OP(SPECIALIZE_ConvertBuiltinSymbolToArithmetic);
PEXPR_FOR_EACH_UNARY_OP(SPECIALIZE_ConvertBuiltinSymbolToArithmetic);
#undef SPECIALIZE_ConvertBuiltinSymbolToArithmetic

template <typename BuiltinSymbol>
constexpr const char* GetBuiltinSymbolDebugString() {
  if constexpr (ConvertBuiltinSymbolToArithmetic<BuiltinSymbol>::convertable) {
    return ConvertBuiltinSymbolToArithmetic<
        BuiltinSymbol>::arithmetic_op_type::Name();
  } else {
    return BuiltinSymbol::Name();
  }
}

template <>
struct TypeImpl<builtin_symbol::Symbol> : public std::monostate {
  using value_type = builtin_symbol::Symbol;

  const char* Name() const { return "builtin_symbol"; }
};

}  // namespace pexpr
