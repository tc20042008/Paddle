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
#include "paddle/cinn/adt/adt.h"
#include "paddle/phi/common/ap/adt.h"
#include "paddle/phi/common/ap/data_type.h"
#include "paddle/pir/include/dialect/pexpr/value.h"

namespace phi {

class DenseTensor;

}

namespace ap::kernel_define {

namespace adt = ::cinn::adt;
using pexpr::ArithmeticType;
using pexpr::PointerType;

using ArgTypeImpl = std::variant<ArithmeticType, PointerType>;

struct ArgType : public ArgTypeImpl {
  using ArgTypeImpl::ArgTypeImpl;
  DEFINE_ADT_VARIANT_METHODS(ArgTypeImpl);

  const char* Name() const {
    return Match([](const auto& impl) { return impl.Name(); });
  }

  template <typename T>
  adt::Result<T> TryGet() const {
    if (!this->template Has<T>()) {
      return adt::errors::TypeError{
          std::string() + "ArgType::TryGet() failed. T: " + typeid(T).name()};
    }
    return this->template Get<T>();
  }

  template <typename T>
  bool IsType() const {
    if constexpr (std::is_pointer_v<T>) {
      const auto& pointer_type = this->template TryGet<pexpr::PointerType>();
      if (pointer_type.HasError()) {
        return false;
      }
      return pointer_type.GetOkValue().template Has<pexpr::CppPointerType<T>>();
    } else {
      const auto& arithmetic_type =
          this->template TryGet<pexpr::ArithmeticType>();
      if (arithmetic_type.HasError()) {
        return false;
      }
      return arithmetic_type.GetOkValue()
          .template Has<pexpr::CppArithmeticType<T>>();
    }
  }
};

struct DefinerRawCtxImpl {
  bool operator==(const DefinerRawCtxImpl& other) { return &other == this; }
};
DEFINE_ADT_RC(DefinerRawCtx, DefinerRawCtxImpl);

template <typename ValueT>
struct DefinerCtxImpl {
  DefinerRawCtx raw_ctx;
  pexpr::Object<ValueT> objects;

  bool operator==(const DefinerCtxImpl& other) const {
    return other.raw_ctx == this->raw_ctx && other.objects == this->objects;
  }
};

template <typename ValueT>
DEFINE_ADT_RC(DefinerCtx, DefinerCtxImpl<ValueT>);

using FuncId = std::string;

struct FuncDeclareImpl {
  FuncId func_id;
  adt::List<ArgType> arg_types;

  bool operator==(const FuncDeclareImpl& other) const {
    return other.func_id == this->func_id && other.arg_types == this->arg_types;
  }
};
DEFINE_ADT_RC(FuncDeclare, FuncDeclareImpl);

struct SourceCodeImpl {
  std::string source_code;

  bool operator==(const SourceCodeImpl& other) const {
    return other.source_code == this->source_code;
  }
};
DEFINE_ADT_RC(SourceCode, SourceCodeImpl);

struct ModuleImpl {
  adt::List<FuncDeclare> func_declares;
  SourceCode source_code;

  bool operator==(const ModuleImpl& other) const {
    return other.func_declares == this->func_declares &&
           other.source_code == this->source_code;
  }
};
DEFINE_ADT_RC(Module, ModuleImpl);

template <typename ValueT>
using CustomValueImpl = std::
    variant<DefinerRawCtx, DefinerCtx<ValueT>, FuncDeclare, SourceCode, Module>;

struct CustomValue : public CustomValueImpl<pexpr::Value<CustomValue>> {
  using CustomValueImpl<pexpr::Value<CustomValue>>::CustomValueImpl;
  DEFINE_ADT_VARIANT_METHODS(CustomValueImpl<pexpr::Value<CustomValue>>);
};

using Val = pexpr::Value<CustomValue>;

using Env = pexpr::Environment<Val>;

using EnvMgr = pexpr::EnvironmentManager<Val>;

template <typename T>
struct GetCustomValueTypeNameHelper;

template <>
struct GetCustomValueTypeNameHelper<DefinerRawCtx> {
  static const char* Call() { return "DefinerRawCtx"; }
};

template <>
struct GetCustomValueTypeNameHelper<DefinerCtx<Val>> {
  static const char* Call() { return "DefinerCtx"; }
};

template <>
struct GetCustomValueTypeNameHelper<FuncDeclare> {
  static const char* Call() { return "FuncDeclare"; }
};

template <>
struct GetCustomValueTypeNameHelper<SourceCode> {
  static const char* Call() { return "SourceCode"; }
};

template <>
struct GetCustomValueTypeNameHelper<Module> {
  static const char* Call() { return "Module"; }
};

template <typename T>
const char* GetCustomValueTypeNameImpl() {
  return GetCustomValueTypeNameHelper<T>::Call();
}

inline const char* GetCustomValueTypeName(const CustomValue& value) {
  return value.Match([](const auto& impl) {
    return GetCustomValueTypeNameImpl<std::decay_t<decltype(impl)>>();
  });
}

template <typename T>
Result<T> CastToCustomValue(const Val& value) {
  if (!value.Has<CustomValue>()) {
    return adt::errors::TypeError{
        std::string() +
        "cast failed. expected type: " + GetCustomValueTypeNameImpl<T>() +
        ", actual type: " + GetBuiltinTypeName(value)};
  }
  const auto& custom_value = value.Get<CustomValue>();
  if (!custom_value.Has<T>()) {
    return adt::errors::TypeError{
        std::string() +
        "cast failed. expected type: " + GetCustomValueTypeNameImpl<T>() +
        ", actual type: " + GetCustomValueTypeName(custom_value)};
  }
  return custom_value.Get<T>();
}

Result<Val> CustomGetAttr(const CustomValue&, const std::string& name);

inline Result<Val> CustomGetItem(const CustomValue&, const Val& idx) {
  return adt::errors::TypeError{"'IndexExprValue' object is not subscriptable"};
}

inline Result<ArgType> CastToArgType(const Val& val) {
  return val.Match(
      [&](const ArithmeticType& atype) -> Result<ArgType> {
        return ArgType{atype};
      },
      [&](const PointerType& ptype) -> Result<ArgType> {
        return ArgType{ptype};
      },
      [&](const auto&) -> Result<ArgType> {
        return TypeError{std::string() +
                         "CastToArgType failed. expected types: "
                         "(ArithmeticType, PointerType), actual type: " +
                         GetBuiltinTypeName(val)};
      });
}

}  // namespace ap::kernel_define
