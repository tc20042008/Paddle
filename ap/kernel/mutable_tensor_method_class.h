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

#include "ap/axpr/data_type_util.h"
#include "ap/axpr/method_class.h"
#include "ap/kernel/mutable_tensor.h"

namespace ap::kernel_dispatch {

using ap::axpr::BuiltinBinaryFuncT;
using ap::axpr::BuiltinFuncType;
using ap::axpr::BuiltinUnaryFuncT;
using ap::axpr::CppDataType;
using ap::axpr::CppPointerType;
using ap::axpr::DataType;
using ap::axpr::DataValue;
using ap::axpr::Method;
using ap::axpr::MethodClass;
using ap::axpr::PointerType;
using ap::axpr::PointerValue;

namespace detail {

template <typename Val>
Result<Val> MutableTensorShapeGetAttr(const MutableTensor<Val>& tensor,
                                      const std::string&) {
  return tensor->dims;
}

template <typename T>
T* GetMutableTensorDataPtr(const ap::axpr::CppDataType<T>&,
                           const MutableTensorData& tensor) {
  return tensor.template data<T>();
}

template <typename Val>
Result<Val> MutableTensorDataGetAttr(const MutableTensor<Val>& tensor,
                                     const std::string&) {
  phi::DataType dtype = tensor->tensor_data.dtype();
  const auto& data_type = ap::axpr::GetDataTypeFromPhiDataType(dtype);
  ADT_RETURN_IF_ERR(data_type);
  return data_type.GetOkValue().Match(
      [&](const adt::Undefined&) -> Result<Val> {
        return TypeError{"dtype is invalid."};
      },
      [&](const auto& impl) -> Result<Val> {
        return PointerValue{GetMutableTensorDataPtr(impl, tensor->tensor_data)};
      });
}

template <typename Val>
using MutableTensorGetAttrT = Result<Val> (*)(const MutableTensor<Val>& tensor,
                                              const std::string&);

template <typename Val>
Result<Val> TensorGetAttr(const MutableTensor<Val>& tensor,
                          const std::string& name) {
  static const std::unordered_map<std::string, MutableTensorGetAttrT<Val>> map{
      {"shape", &MutableTensorShapeGetAttr<Val>},
      {"data_ptr", &MutableTensorDataGetAttr<Val>},
  };
  const auto& iter = map.find(name);
  if (iter == map.end()) {
    return AttributeError{std::string("'Tensor' has no attribute '") + name +
                          "'"};
  }
  return iter->second(tensor, name);
}

}  // namespace detail

template <typename ValueT>
struct MutableTensorMethodClass {
  using Self = MutableTensorMethodClass;

  template <typename BuiltinUnarySymbol>
  static std::optional<BuiltinUnaryFuncT<ValueT>> GetBuiltinUnaryFunc() {
    return std::nullopt;
  }

  template <typename BultinBinarySymbol>
  static std::optional<BuiltinBinaryFuncT<ValueT>> GetBuiltinBinaryFunc() {
    if constexpr (std::is_same_v<BultinBinarySymbol,
                                 ap::axpr::builtin_symbol::GetAttr>) {
      return &Self::GetAttr;
    }
    return std::nullopt;
  }

  static adt::Result<ValueT> GetAttr(const ValueT& obj_val,
                                     const ValueT& attr_name_val) {
    const auto& opt_obj =
        MethodClass<ValueT>::template TryGet<MutableTensor<Val>>(obj_val);
    ADT_RETURN_IF_ERR(opt_obj);
    const auto& obj = opt_obj.GetOkValue();
    const auto& opt_attr_name =
        MethodClass<ValueT>::template TryGet<std::string>(attr_name_val);
    ADT_RETURN_IF_ERR(opt_attr_name);
    const auto& attr_name = opt_attr_name.GetOkValue();
    return detail::TensorGetAttr<Val>(obj, attr_name);
  }
};

}  // namespace ap::kernel_dispatch

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, ap::kernel_dispatch::MutableTensor<ValueT>> {
  using method_class = ap::kernel_dispatch::MutableTensorMethodClass<ValueT>;

  template <typename BuiltinUnarySymbol>
  static std::optional<BuiltinUnaryFuncT<ValueT>> GetBuiltinUnaryFunc() {
    return method_class::template GetBuiltinUnaryFunc<BuiltinUnarySymbol>();
  }

  template <typename BultinBinarySymbol>
  static std::optional<BuiltinBinaryFuncT<ValueT>> GetBuiltinBinaryFunc() {
    return method_class::template GetBuiltinBinaryFunc<BultinBinarySymbol>();
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT,
                       TypeImpl<ap::kernel_dispatch::MutableTensor<ValueT>>>
    : public EmptyMethodClass<ValueT> {};

}  // namespace ap::axpr
