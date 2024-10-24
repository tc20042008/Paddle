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

#include "ap/axpr/adt.h"
#include "ap/axpr/const_std_vector_ptr.h"
#include "ap/axpr/constants.h"
#include "ap/axpr/method_class.h"

namespace ap::axpr {

template <typename ValueT, typename T>
struct StdVectorPtrMethodClass {
  using This = StdVectorPtrMethodClass;
  using Self = std::vector<T>*;

  adt::Result<ValueT> ToString(const Self& self) {
    std::ostringstream ss;
    const void* ptr = self;
    ss << "<" << axpr::TypeImpl<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  template <typename BuiltinUnarySymbol>
  static std::optional<BuiltinUnaryFuncT<ValueT>> GetBuiltinUnaryFunc() {
    return std::nullopt;
  }

  template <typename BultinBinarySymbol>
  static std::optional<BuiltinBinaryFuncT<ValueT>> GetBuiltinBinaryFunc() {
    if constexpr (std::is_same_v<BultinBinarySymbol, builtin_symbol::GetItem>) {
      return &This::GetItem;
    }
    return std::nullopt;
  }

  static adt::Result<ValueT> GetItem(const ValueT& vect_value,
                                     const ValueT& idx) {
    const auto& opt_vect =
        MethodClass<ValueT>::template TryGet<std::vector<T>*>(vect_value);
    ADT_RETURN_IF_ERR(opt_vect);
    const auto& vect = opt_vect.GetOkValue();
    return idx.Match(
        [&](int64_t index) -> Result<ValueT> {
          if (index < 0) {
            index += vect->size();
          }
          if (index >= 0 && index < vect->size()) {
            return CastItem(vect->at(index));
          }
          return adt::errors::IndexError{"vector index out of range"};
        },
        [&](const auto&) -> Result<ValueT> {
          return adt::errors::TypeError{
              std::string() + "vector indices must be integers, not " +
              MethodClass<ValueT>::Name(idx)};
        });
  }

  static adt::Result<ValueT> CastItem(const T& elem) {
    if constexpr (std::is_integral_v<T>) {
      return static_cast<int64_t>(elem);
    } else if constexpr (std::is_floating_point<T>::value) {
      return static_cast<double>(elem);
    } else {
      return elem;
    }
  }
};

template <typename ValueT, typename T>
struct MethodClassImpl<ValueT, std::vector<T>*>
    : public StdVectorPtrMethodClass<ValueT, T> {};

template <typename ValueT, typename T>
struct MethodClassImpl<ValueT, TypeImpl<std::vector<T>*>>
    : public EmptyMethodClass<ValueT> {};

}  // namespace ap::axpr
