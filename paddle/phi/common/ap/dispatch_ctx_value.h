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
#include "paddle/phi/common/ap/adt.h"
#include "paddle/phi/common/ap/data_type.h"
#include "paddle/phi/common/ap/define_ctx_value.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/pir/include/dialect/pexpr/value.h"

namespace phi {

class DenseTensor;

}

namespace ap::kernel_dispatch {

namespace adt = ::cinn::adt;

using kernel_define::ArgType;

using CppValueImpl = std::variant<
#define MAKE_ARG_VALUE_ALTERNATIVE(cpp_type, enum_type) \
  cpp_type, const cpp_type, cpp_type*, const cpp_type*,
    PD_FOR_EACH_DATA_TYPE(MAKE_ARG_VALUE_ALTERNATIVE) void*,
    const void*
#undef MAKE_ARG_VALUE_ALTERNATIVE
    >;

struct CppValue : public CppValueImpl {
  using CppValueImpl::CppValueImpl;
  DEFINE_ADT_VARIANT_METHODS(CppValueImpl);
};

inline ArgType GetArgType(const CppValue& cpp_value) {
  return cpp_value.Match([](auto impl) -> ArgType {
    return kernel_define::CppArgType<decltype(impl)>{};
  });
}

struct TypedBufferImpl {
  void* buffer;
  phi::DataType dtype;
  size_t size;

  bool operator==(const TypedBufferImpl& other) const {
    return other.buffer == this->buffer && other.dtype == this->dtype &&
           other.size == this->size;
  }
};
DEFINE_ADT_RC(TypedBuffer, TypedBufferImpl);

using ConstTensorDataImpl = std::variant<const phi::DenseTensor*, TypedBuffer>;
struct ConstTensorData : public ConstTensorDataImpl {
  using ConstTensorDataImpl::ConstTensorDataImpl;
  DEFINE_ADT_VARIANT_METHODS(ConstTensorDataImpl);

  template <typename T>
  const T* data() const {
    return Match(
        [](const phi::DenseTensor* tensor) -> const T* {
          return reinterpret_cast<const T*>(tensor->data());
        },
        [](const TypedBuffer& buffer) -> const T* {
          return reinterpret_cast<T*>(buffer->buffer);
        });
  }
  const void* data() const {
    return Match(
        [](const phi::DenseTensor* tensor) -> const void* {
          return tensor->data();
        },
        [](const TypedBuffer& buffer) -> const void* {
          return buffer->buffer;
        });
  }
  phi::DataType dtype() const {
    return Match([](const phi::DenseTensor* tensor) { return tensor->dtype(); },
                 [](const TypedBuffer& buffer) { return buffer->dtype; });
  }
};

template <typename ValueT>
struct ConstTensorImpl {
  ConstTensorData tensor_data;
  adt::List<ValueT> dims;

  bool operator==(const ConstTensorImpl& other) const {
    return other.tensor_data == this->tensor_data && other.dims == this->dims;
  }
};

template <typename ValueT>
DEFINE_ADT_RC(ConstTensor, ConstTensorImpl<ValueT>);

using MutableTensorDataImpl = std::variant<phi::DenseTensor*, TypedBuffer>;
struct MutableTensorData : public MutableTensorDataImpl {
  using MutableTensorDataImpl::MutableTensorDataImpl;
  DEFINE_ADT_VARIANT_METHODS(MutableTensorDataImpl);

  template <typename T>
  T* data() const {
    return Match(
        [](phi::DenseTensor* tensor) -> T* {
          return reinterpret_cast<T*>(tensor->data());
        },
        [](const TypedBuffer& buffer) -> T* {
          return reinterpret_cast<T*>(buffer->buffer);
        });
  }
  void* data() const {
    return Match(
        [](phi::DenseTensor* tensor) -> void* { return tensor->data(); },
        [](const TypedBuffer& buffer) -> void* { return buffer->buffer; });
  }
  phi::DataType dtype() const {
    return Match([](phi::DenseTensor* tensor) { return tensor->dtype(); },
                 [](const TypedBuffer& buffer) { return buffer->dtype; });
  }
};

template <typename ValueT>
struct MutableTensorImpl {
  MutableTensorData tensor_data;
  adt::List<ValueT> dims;

  bool operator==(const MutableTensorImpl& other) const {
    return other.tensor_data == this->tensor_data && other.dims == this->dims;
  }
};
template <typename ValueT>
DEFINE_ADT_RC(MutableTensor, MutableTensorImpl<ValueT>);

class CudaModule {
 public:
  CudaModule(const CudaModule&) = delete;
  CudaModule(CudaModule&&) = delete;
  virtual ~CudaModule() = default;

  virtual adt::Result<adt::Ok> LaunchCudaKernel(
      const std::string& func_name,
      int64_t num_blocks,
      int64_t num_threads,
      const std::vector<void*>& args) = 0;

 protected:
  CudaModule() = default;
};

template <typename ValueT>
struct DispatchRawContextImpl {
  adt::List<ValueT> inputs;
  adt::List<ValueT> outputs;
  std::shared_ptr<CudaModule> cuda_module;
  std::unordered_map<std::string, adt::List<kernel_define::ArgType>>
      func_name2arg_types;

  bool operator==(const DispatchRawContextImpl& other) const {
    return &other == this;
  }

  Result<adt::Ok> LaunchCudaKernel(
      const std::string& func_name,
      int64_t num_blocks,
      int64_t num_threads,
      const adt::List<CppValue>& kernel_args) const;
};

template <typename ValueT>
DEFINE_ADT_RC(DispatchRawContext, DispatchRawContextImpl<ValueT>);

template <typename ValueT>
struct DispatchContextImpl {
  DispatchRawContext<ValueT> raw_ctx;
  pexpr::Object<ValueT> data;

  bool operator==(const DispatchContextImpl& other) const {
    return &other == this;
  }
};

template <typename ValueT>
DEFINE_ADT_RC(DispatchContext, DispatchContextImpl<ValueT>);

template <typename ValueT>
using CustomValueImpl = std::variant<ArgType,
                                     CppValue,
                                     ConstTensor<ValueT>,
                                     MutableTensor<ValueT>,
                                     DispatchRawContext<ValueT>,
                                     DispatchContext<ValueT>>;

struct CustomValue : public CustomValueImpl<pexpr::Value<CustomValue>> {
  using CustomValueImpl<pexpr::Value<CustomValue>>::CustomValueImpl;
  DEFINE_ADT_VARIANT_METHODS(CustomValueImpl<pexpr::Value<CustomValue>>);
};

using Val = pexpr::Value<CustomValue>;

using Env = pexpr::Environment<Val>;

using EnvMgr = pexpr::EnvironmentManager<Val>;

template <typename T>
struct GetCppValueTypeNameHelper;

#define SPECIALIZE_GET_CUSTOM_VALUE_TYPE_NAME_IMPL(type) \
  template <>                                            \
  struct GetCppValueTypeNameHelper<type> {               \
    static const char* Call() { return #type; }          \
  };
#define MAKE_CPP_TYPE_CASE(cpp_type, enum_type)               \
  SPECIALIZE_GET_CUSTOM_VALUE_TYPE_NAME_IMPL(cpp_type);       \
  SPECIALIZE_GET_CUSTOM_VALUE_TYPE_NAME_IMPL(const cpp_type); \
  SPECIALIZE_GET_CUSTOM_VALUE_TYPE_NAME_IMPL(cpp_type*);      \
  SPECIALIZE_GET_CUSTOM_VALUE_TYPE_NAME_IMPL(const cpp_type*);
PD_FOR_EACH_DATA_TYPE(MAKE_CPP_TYPE_CASE)
#undef MAKE_CPP_TYPE_CASE

template <>
struct GetCppValueTypeNameHelper<void*> {
  static const char* Call() { return "void*"; }
};

template <>
struct GetCppValueTypeNameHelper<const void*> {
  static const char* Call() { return "const void*"; }
};

template <typename T>
inline const char* GetCppValueTypeNameImpl() {
  return GetCppValueTypeNameHelper<std::decay_t<T>>::Call();
}

inline const char* GetCppValueTypeName(const CppValue& val) {
  return val.Match(
      [](auto impl) { return GetCppValueTypeNameImpl<decltype(impl)>(); });
}

template <typename T>
struct GetCustomValueTypeNameHelper;

#define SPECIALIZE_GET_CUSTOM_VALUE_TYPE_NAME_IMPL(type) \
  template <>                                            \
  struct GetCustomValueTypeNameHelper<type> {            \
    static const char* Call() { return #type; }          \
  };

SPECIALIZE_GET_CUSTOM_VALUE_TYPE_NAME_IMPL(ArgType);
SPECIALIZE_GET_CUSTOM_VALUE_TYPE_NAME_IMPL(CppValue);
SPECIALIZE_GET_CUSTOM_VALUE_TYPE_NAME_IMPL(ConstTensor<Val>);
SPECIALIZE_GET_CUSTOM_VALUE_TYPE_NAME_IMPL(MutableTensor<Val>);
SPECIALIZE_GET_CUSTOM_VALUE_TYPE_NAME_IMPL(DispatchRawContext<Val>);
SPECIALIZE_GET_CUSTOM_VALUE_TYPE_NAME_IMPL(DispatchContext<Val>);
#undef SPECIALIZE_GET_CUSTOM_VALUE_TYPE_NAME_IMPL

template <typename T>
const char* GetCustomValueTypeNameImpl() {
  return GetCustomValueTypeNameHelper<std::decay_t<T>>::Call();
}

inline const char* GetCustomValueTypeName(const CustomValue& value) {
  return value.Match([](const auto& impl) {
    return GetCustomValueTypeNameImpl<std::decay_t<decltype(impl)>>();
  });
}

template <typename T>
Result<T> CastToCustomValue(const Val& value) {
  if (!value.Has<CustomValue>()) {
    if (value.Has<bool>()) {
      if constexpr (std::is_same_v<std::decay_t<T>, CppValue>) {
        return CppValue{value.Get<bool>()};
      }
    }
    if (value.Has<int64_t>()) {
      if constexpr (std::is_same_v<std::decay_t<T>, CppValue>) {
        return CppValue{value.Get<int64_t>()};
      }
    }
    return TypeError{std::string() + "cast failed. expected type: " +
                     GetCustomValueTypeNameImpl<T>() +
                     ", actual type: " + GetBuiltinTypeName(value)};
  }
  const auto& custom_value = value.Get<CustomValue>();
  if (!custom_value.Has<T>()) {
    return TypeError{std::string() + "cast failed. expected type: " +
                     GetCustomValueTypeNameImpl<T>() +
                     ", actual type: " + GetCustomValueTypeName(custom_value)};
  }
  return custom_value.Get<T>();
}

template <typename T>
Result<T> CastToCppValue(const Val& value) {
  const auto& opt_cpp_value = CastToCustomValue<CppValue>(value);
  ADT_RETURN_IF_ERROR(opt_cpp_value);
  const auto& cpp_value = opt_cpp_value.GetOkValue();
  if (!cpp_value.Has<T>()) {
    return TypeError{std::string() + "cast failed. expected type: " +
                     GetCppValueTypeNameImpl<T>() +
                     ", actual type: " + GetCppValueTypeName(cpp_value)};
  }
  return cpp_value.Get<T>();
}

Result<Val> CustomGetAttr(const CustomValue& custom_value,
                          const std::string& name);

inline Result<Val> CustomGetItem(const CustomValue&, const Val& idx) {
  return TypeError{"'IndexExprValue' object is not subscriptable"};
}

}  // namespace ap::kernel_dispatch
