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

template <typename T>
struct GetCppTypeNameHelper;

#define SPECIALIZE_GET_CPP_TYPE_NAME(cpp_type, enum_type)        \
  template <>                                                    \
  struct GetCppTypeNameHelper<cpp_type> {                        \
    static const char* Call() { return #cpp_type; }              \
  };                                                             \
  template <>                                                    \
  struct GetCppTypeNameHelper<const cpp_type> {                  \
    static const char* Call() { return "const " #cpp_type; }     \
  };                                                             \
  template <>                                                    \
  struct GetCppTypeNameHelper<cpp_type*> {                       \
    static const char* Call() { return #cpp_type "*"; }          \
  };                                                             \
  template <>                                                    \
  struct GetCppTypeNameHelper<const cpp_type*> {                 \
    static const char* Call() { return "const " #cpp_type "*"; } \
  };
PD_FOR_EACH_DATA_TYPE(SPECIALIZE_GET_CPP_TYPE_NAME);
#undef SPECIALIZE_GET_CPP_TYPE_NAME

template <>
struct GetCppTypeNameHelper<void*> {
  static const char* Call() { return "void*"; }
};

template <>
struct GetCppTypeNameHelper<const void*> {
  static const char* Call() { return "const void*"; }
};

template <typename T>
struct CppArgType {
  using type = T;
  bool operator==(const CppArgType& other) const { return true; }
  const char* name() const { return GetCppTypeNameHelper<T>::Call(); }
};

// clang-format off
using ArgTypeImpl = std::variant<
#define MAKE_CPP_TYPE_ALTERNATIVE(cpp_type, enum_type)    \
    CppArgType<cpp_type>,                                 \
    CppArgType<const cpp_type>,                           \
    CppArgType<cpp_type*>,                                \
    CppArgType<const cpp_type*>,
    PD_FOR_EACH_DATA_TYPE(MAKE_CPP_TYPE_ALTERNATIVE)
#undef MAKE_CPP_TYPE_ALTERNATIVE
    CppArgType<void*>,
    CppArgType<const void*>>;
// clang-format on

struct ArgType : public ArgTypeImpl {
  using ArgTypeImpl::ArgTypeImpl;
  DEFINE_ADT_VARIANT_METHODS(ArgTypeImpl);

  const char* name() const {
    return Match([](const auto& impl) { return impl.name(); });
  }

  static adt::Result<ArgType> MakeFromPhiDataType(::phi::DataType);

  ArgType RemoveConst() const;
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
using CustomValueImpl = std::variant<DefinerRawCtx,
                                     DefinerCtx<ValueT>,
                                     ArgType,
                                     FuncDeclare,
                                     SourceCode,
                                     Module>;

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
struct GetCustomValueTypeNameHelper<ArgType> {
  static const char* Call() { return "ArgType"; }
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

}  // namespace ap::kernel_define
