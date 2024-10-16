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

#include <experimental/type_traits>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include "ap/axpr/adt.h"
#include "ap/axpr/builtin_func_type.h"
#include "ap/axpr/constants.h"
#include "ap/axpr/type.h"

namespace ap::axpr {

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
  static constexpr std::optional<BuiltinUnaryFuncT<ValueT>>
  GetBuiltinUnaryFunc() {
    return std::nullopt;
  }

  template <typename BultinBinarySymbol>
  static constexpr std::optional<BuiltinBinaryFuncT<ValueT>>
  GetBuiltinBinaryFunc() {
    return std::nullopt;
  }
};

template <typename ValueT, typename T>
struct MethodClassImpl;

namespace detail {

template <typename ValueT, typename T, typename BuiltinSymbol>
struct BuiltinMethodHelperImpl;

#define SPECIALIZE_BuiltinMethodHelperImpl(symbol_name, op)                  \
  template <typename ValueT, typename T>                                     \
  struct BuiltinMethodHelperImpl<ValueT, T, builtin_symbol::symbol_name> {   \
    using This =                                                             \
        BuiltinMethodHelperImpl<ValueT, T, builtin_symbol::symbol_name>;     \
                                                                             \
    template <typename ObjT>                                                 \
    using UnaryMethodRetT =                                                  \
        decltype(std::declval<MethodClassImpl<ValueT, ObjT>&>().symbol_name( \
            std::declval<const ObjT&>()));                                   \
                                                                             \
    static constexpr bool HasUnaryMethod() {                                 \
      return builtin_symbol::symbol_name::num_operands == 1 &&               \
             std::experimental::is_detected_v<UnaryMethodRetT, T>;           \
    }                                                                        \
                                                                             \
    static adt::Result<ValueT> UnaryCall(const T& obj) {                     \
      if constexpr (This::HasUnaryMethod()) {                                \
        return MethodClassImpl<ValueT, T>{}.symbol_name(obj);                \
      } else {                                                               \
        return adt::errors::RuntimeError{"`" #symbol_name                    \
                                         "` method not found."};             \
      }                                                                      \
    }                                                                        \
                                                                             \
    template <typename ObjT>                                                 \
    using BinaryMethodRetT =                                                 \
        decltype(std::declval<MethodClassImpl<ValueT, ObjT>&>().symbol_name( \
            std::declval<const ObjT&>(), std::declval<const ValueT&>()));    \
                                                                             \
    static constexpr bool HasBinaryMethod() {                                \
      return builtin_symbol::symbol_name::num_operands == 2 &&               \
             std::experimental::is_detected_v<BinaryMethodRetT, T>;          \
    }                                                                        \
                                                                             \
    static adt::Result<ValueT> BinaryCall(const T& obj, const ValueT& arg) { \
      if constexpr (This::HasBinaryMethod()) {                               \
        return MethodClassImpl<ValueT, T>{}.symbol_name(obj, arg);           \
      } else {                                                               \
        return adt::errors::RuntimeError{"`" #symbol_name                    \
                                         "` method not found."};             \
      }                                                                      \
    }                                                                        \
  };

AXPR_FOR_EACH_SYMBOL_OP(SPECIALIZE_BuiltinMethodHelperImpl)

#undef SPECIALIZE_BuiltinMethodHelperImpl

template <typename VariantT, typename T>
struct DirectAlternative {
  static adt::Result<T> TryGet(const VariantT& val) {
    if (val.template Has<T>()) {
      return val.template Get<T>();
    }
    return adt::errors::TypeError{"cast failed."};
  }
};

template <typename ValueT, typename T>
struct IndirectAlternative {
  static adt::Result<T> TryGet(const ValueT& val) {
    using TypeT = typename TypeTrait<ValueT>::TypeT;
    ADT_LET_CONST_REF(type, DirectAlternative<ValueT, TypeT>::TryGet(val));
    return DirectAlternative<TypeT, T>::TryGet(type);
  }
};

template <typename ValueT,
          typename T,
          typename BuiltinSymbol,
          template <typename, typename>
          class Alternative>
struct BuiltinMethodHelper {
  using This = BuiltinMethodHelper;
  using Impl = BuiltinMethodHelperImpl<ValueT, T, BuiltinSymbol>;

  static constexpr bool HasUnaryMethod() { return Impl::HasUnaryMethod(); }

  static constexpr BuiltinUnaryFuncT<ValueT> GetBuiltinUnaryMethod() {
    return &This::MakeBuiltinUnaryFunc<&Impl::UnaryCall>;
  }

  static std::optional<BuiltinUnaryFuncT<ValueT>> GetBuiltinUnaryFunc() {
    static const MethodClassImpl<ValueT, T>
        detect_specialization_of_method_class_impl;
    (void)detect_specialization_of_method_class_impl;
    if constexpr (BuiltinMethodHelperImpl<ValueT, T, BuiltinSymbol>::
                      HasUnaryMethod()) {
      return &This::MakeBuiltinUnaryFunc<
          &BuiltinMethodHelperImpl<ValueT, T, BuiltinSymbol>::UnaryCall>;
    } else if constexpr (HasDefaultUnaryMethod()) {
      return MethodClassImpl<ValueT,
                             T>::template GetBuiltinUnaryFunc<BuiltinSymbol>();
    } else {
      return std::nullopt;
    }
  }

  template <typename ObjT>
  using UnaryMethodRetT =
      decltype(MethodClassImpl<ValueT, ObjT>::template GetBuiltinUnaryFunc<
               BuiltinSymbol>());

  static constexpr bool HasDefaultUnaryMethod() {
    return std::experimental::is_detected_v<UnaryMethodRetT, T>;
  }

  static std::optional<BuiltinBinaryFuncT<ValueT>> GetBuiltinBinaryFunc() {
    static const MethodClassImpl<ValueT, T>
        detect_specialization_of_method_class_impl;
    (void)detect_specialization_of_method_class_impl;
    if constexpr (BuiltinMethodHelperImpl<ValueT, T, BuiltinSymbol>::
                      HasBinaryMethod()) {
      return &This::MakeBuiltinBinaryFunc<
          &BuiltinMethodHelperImpl<ValueT, T, BuiltinSymbol>::BinaryCall>;
    } else if constexpr (HasDefaultBinaryMethod()) {
      return MethodClassImpl<ValueT,
                             T>::template GetBuiltinBinaryFunc<BuiltinSymbol>();
    } else {
      return std::nullopt;
    }
  }

  template <typename ObjT>
  using BinaryMethodRetT =
      decltype(MethodClassImpl<ValueT, ObjT>::template GetBuiltinBinaryFunc<
               BuiltinSymbol>());

  static constexpr bool HasDefaultBinaryMethod() {
    return std::experimental::is_detected_v<BinaryMethodRetT, T>;
  }

  template <adt::Result<ValueT> (*UnaryFunc)(const T&)>
  static adt::Result<ValueT> MakeBuiltinUnaryFunc(const ValueT& obj_val) {
    ADT_LET_CONST_REF(obj, Alternative<ValueT, T>::TryGet(obj_val));
    const auto& ret = UnaryFunc(obj);
    return ret;
  }

  template <adt::Result<ValueT> (*BinaryFunc)(const T&, const ValueT&)>
  static adt::Result<ValueT> MakeBuiltinBinaryFunc(const ValueT& obj_val,
                                                   const ValueT& arg) {
    ADT_LET_CONST_REF(obj, Alternative<ValueT, T>::TryGet(obj_val));
    return BinaryFunc(obj, arg);
  }
};

}  // namespace detail

template <typename ValueT>
struct MethodClass {
  using This = MethodClass;

  template <typename T>
  static adt::Result<T> TryGet(const ValueT& val) {
    if (val.template Has<T>()) {
      return val.template Get<T>();
    }
    return adt::errors::TypeError{
        std::string() + "cast failed. expected type: " + TypeImpl<T>{}.Name() +
        ", actual type: " + This::Name(val)};
  }

  static const char* Name(const ValueT& val) {
    return val.Match([](const auto& impl) -> const char* {
      using T = std::decay_t<decltype(impl)>;
      return TypeImpl<T>{}.Name();
    });
  }

  static BuiltinUnaryFuncT<ValueT> ToString(const ValueT& val) {
    using S = builtin_symbol::ToString;
    return val.Match([](const auto& impl) -> BuiltinUnaryFuncT<ValueT> {
      using T = std::decay_t<decltype(impl)>;
      if constexpr (IsType<T>()) {
        return impl.Match([](const auto& type_impl)
                              -> BuiltinUnaryFuncT<ValueT> {
          using TT = std::decay_t<decltype(type_impl)>;
          using Helper = detail::
              BuiltinMethodHelper<ValueT, TT, S, detail::IndirectAlternative>;
          if constexpr (Helper::HasUnaryMethod()) {
            return Helper::GetBuiltinUnaryMethod();
          } else {
            return &This::DefaultTypeToString<TT>;
          }
        });
      } else {
        using Helper = detail::
            BuiltinMethodHelper<ValueT, T, S, detail::DirectAlternative>;
        if constexpr (Helper::HasUnaryMethod()) {
          return Helper::GetBuiltinUnaryMethod();
        } else {
          return &This::DefaultInstanceToString<T>;
        }
      }
    });
  }

  template <typename TT>
  static adt::Result<ValueT> DefaultTypeToString(const ValueT& val) {
    std::ostringstream ss;
    ss << "<class '" << TT{}.Name() << "'>";
    return ss.str();
  }

  template <typename T>
  static adt::Result<ValueT> DefaultInstanceToString(const ValueT& val) {
    std::ostringstream ss;
    ADT_LET_CONST_REF(impl, This::TryGet<T>(val));
    // please implement MethodClassImpl<ValueT, T>::ToString if T is not defined
    // by DEFINE_ADT_RC.
    const void* ptr = impl.__adt_rc_shared_ptr_raw_ptr();
    ss << "<" << This::Name(val) << " object at " << ptr << ">";
    return ss.str();
  }

  template <typename BultinUnarySymbol>
  static std::optional<BuiltinUnaryFuncT<ValueT>> GetBuiltinUnaryFunc(
      const ValueT& val) {
    using S = BultinUnarySymbol;
    return val.Match([](const auto& impl)
                         -> std::optional<BuiltinUnaryFuncT<ValueT>> {
      using T = std::decay_t<decltype(impl)>;
      if constexpr (IsType<T>()) {
        return impl.Match([](const auto& type_impl)
                              -> std::optional<BuiltinUnaryFuncT<ValueT>> {
          using TT = std::decay_t<decltype(type_impl)>;
          using Helper = detail::
              BuiltinMethodHelper<ValueT, TT, S, detail::IndirectAlternative>;
          return Helper::GetBuiltinUnaryFunc();
        });
      } else {
        using Helper = detail::
            BuiltinMethodHelper<ValueT, T, S, detail::DirectAlternative>;
        return Helper::GetBuiltinUnaryFunc();
      }
    });
  }

  template <typename BultinBinarySymbol>
  static std::optional<BuiltinBinaryFuncT<ValueT>> GetBuiltinBinaryFunc(
      const ValueT& val) {
    using S = BultinBinarySymbol;
    return val.Match([](const auto& impl)
                         -> std::optional<BuiltinBinaryFuncT<ValueT>> {
      using T = std::decay_t<decltype(impl)>;
      if constexpr (IsType<T>()) {
        return impl.Match([](const auto& type_impl)
                              -> std::optional<BuiltinBinaryFuncT<ValueT>> {
          using TT = std::decay_t<decltype(type_impl)>;
          using Helper = detail::
              BuiltinMethodHelper<ValueT, TT, S, detail::IndirectAlternative>;
          return Helper::GetBuiltinBinaryFunc();
        });
      } else {
        using Helper = detail::
            BuiltinMethodHelper<ValueT, T, S, detail::DirectAlternative>;
        return Helper::GetBuiltinBinaryFunc();
      }
    });
  }
};

template <typename ValueT, typename T>
using __AltT = decltype(std::declval<ValueT&>().template Get<T>());

template <typename T, typename ValueT>
adt::Result<T> TryGetAlternative(const ValueT& val) {
  if constexpr (std::experimental::is_detected_v<__AltT, ValueT, T>) {
    return MethodClass<ValueT>::template TryGet<T>(val);
  } else {
    return detail::IndirectAlternative<ValueT, T>::TryGet(val);
  }
}

template <typename T, typename ValueT>
adt::Result<T> TryGetImpl(const ValueT& val) {
  return TryGetAlternative<T, ValueT>(val);
}

template <typename ValueT>
const char* GetTypeName(const ValueT& val) {
  return MethodClass<ValueT>::Name(val);
}

}  // namespace ap::axpr
