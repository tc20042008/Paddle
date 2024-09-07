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
#include "paddle/pir/include/dialect/pexpr/constants.h"
#include "paddle/pir/include/dialect/pexpr/data_value.h"
#include "paddle/pir/include/dialect/pexpr/method_class.h"

namespace pexpr {

template <typename ValueT>
struct ListMethodClass {
  using Self = ListMethodClass;

  static const char* Name() { return "list"; }

  template <typename BuiltinUnarySymbol>
  static std::optional<BuiltinUnaryFuncT<ValueT>> GetBuiltinUnaryFunc() {
    return std::nullopt;
  }

  template <typename BultinBinarySymbol>
  static std::optional<BuiltinBinaryFuncT<ValueT>> GetBuiltinBinaryFunc() {
    if constexpr (std::is_same_v<BultinBinarySymbol, builtin_symbol::GetItem>) {
      return &Self::GetItem;
    }
    return std::nullopt;
  }

  static adt::Result<ValueT> GetItem(const ValueT& lst_value,
                                     const ValueT& idx) {
    const auto& opt_lst =
        MethodClass<ValueT>::template TryGet<adt::List<ValueT>>(lst_value);
    ADT_RETURN_IF_ERROR(opt_lst);
    const auto& lst = opt_lst.GetOkValue();
    return idx.Match(
        [&](const DataValue& arithmetic_idx) -> Result<ValueT> {
          const auto& int64_idx =
              arithmetic_idx.StaticCastTo(CppDataType<int64_t>{});
          ADT_RETURN_IF_ERROR(int64_idx);
          const auto& opt_index =
              int64_idx.GetOkValue().template TryGet<int64_t>();
          ADT_RETURN_IF_ERROR(opt_index);
          int64_t index = opt_index.GetOkValue();
          if (index < 0) {
            index += lst->size();
          }
          if (index >= 0 && index < lst->size()) {
            return lst->at(index);
          }
          return adt::errors::IndexError{"list index out of range"};
        },
        [&](const auto&) -> Result<ValueT> {
          return adt::errors::TypeError{std::string() +
                                        "list indices must be integers, not " +
                                        MethodClass<ValueT>::Name(idx)};
        });
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, adt::List<ValueT>> {
  using method_class = ListMethodClass<ValueT>;

  static const char* Name() { return method_class::Name(); }

  template <typename BuiltinUnarySymbol>
  static std::optional<BuiltinUnaryFuncT<ValueT>> GetBuiltinUnaryFunc() {
    return method_class::template GetBuiltinUnaryFunc<BuiltinUnarySymbol>();
  }

  template <typename BultinBinarySymbol>
  static std::optional<BuiltinBinaryFuncT<ValueT>> GetBuiltinBinaryFunc() {
    return method_class::template GetBuiltinBinaryFunc<BultinBinarySymbol>();
  }
};

}  // namespace pexpr
