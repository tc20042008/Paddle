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

#include <optional>
#include <string>
#include <type_traits>
#include "paddle/pir/include/dialect/pexpr/adt.h"
#include "paddle/pir/include/dialect/pexpr/builtin_func_type.h"
#include "paddle/pir/include/dialect/pexpr/constants.h"

namespace pexpr {

template <typename ValueT>
using BuiltinUnaryFuncT = adt::Result<ValueT> (*)(const ValueT&);

template <typename ValueT, BuiltinFuncType<ValueT> BuiltinFunc>
adt::Result<ValueT> UnaryFuncReturnCapturedValue(const ValueT&) {
  return BuiltinFunc;
}

template <typename ValueT>
using BuiltinBinaryFuncT = adt::Result<ValueT> (*)(const ValueT&,
                                                   const ValueT&);

template <typename ValueT>
struct EmptyMethodClass {
  template <typename BuiltinUnarySymbol>
  static std::optional<BuiltinUnaryFuncT<ValueT>> GetBuiltinUnaryFunc() {
    return std::nullopt;
  }

  template <typename BultinBinarySymbol>
  static std::optional<BuiltinBinaryFuncT<ValueT>> GetBuiltinBinaryFunc() {
    return std::nullopt;
  }
};

template <typename ValueT, typename T>
struct MethodClassImpl;

template <typename ValueT>
struct MethodClass {
  using Self = MethodClass;

  template <typename T>
  static adt::Result<T> TryGet(const ValueT& val) {
    if (val.template Has<T>()) {
      return val.template Get<T>();
    }
    return adt::errors::TypeError{
        std::string() + "cast failed. expected type: " + TypeImpl<T>{}.Name() +
        ", actual type: " + Self::Name(val)};
  }

  static const char* Name(const ValueT& val) {
    return val.Match([](const auto& impl) -> const char* {
      using T = std::decay_t<decltype(impl)>;
      return TypeImpl<T>{}.Name();
    });
  }

  template <typename BultinUnarySymbol>
  static std::optional<BuiltinUnaryFuncT<ValueT>> GetBuiltinUnaryFunc(
      const ValueT& val) {
    return val.Match(
        [](const auto& impl) -> std::optional<BuiltinUnaryFuncT<ValueT>> {
          using T = std::decay_t<decltype(impl)>;
          if constexpr (IsType<T>()) {
            return impl.Match([](const auto& type_impl)
                                  -> std::optional<BuiltinUnaryFuncT<ValueT>> {
              using TT = std::decay_t<decltype(type_impl)>;
              return MethodClassImpl<ValueT, TT>::template GetBuiltinUnaryFunc<
                  BultinUnarySymbol>();
            });
          } else {
            return MethodClassImpl<ValueT, T>::template GetBuiltinUnaryFunc<
                BultinUnarySymbol>();
          }
        });
  }

  template <typename BultinBinarySymbol>
  static std::optional<BuiltinBinaryFuncT<ValueT>> GetBuiltinBinaryFunc(
      const ValueT& val) {
    return val.Match(
        [](const auto& impl) -> std::optional<BuiltinBinaryFuncT<ValueT>> {
          using T = std::decay_t<decltype(impl)>;
          if constexpr (IsType<T>()) {
            return impl.Match([](const auto& type_impl)
                                  -> std::optional<BuiltinBinaryFuncT<ValueT>> {
              using TT = std::decay_t<decltype(type_impl)>;
              return MethodClassImpl<ValueT, TT>::template GetBuiltinBinaryFunc<
                  BultinBinarySymbol>();
            });
          } else {
            return MethodClassImpl<ValueT, T>::template GetBuiltinBinaryFunc<
                BultinBinarySymbol>();
          }
        });
  }
};

}  // namespace pexpr
