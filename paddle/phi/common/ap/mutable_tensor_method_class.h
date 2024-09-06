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

#include "paddle/phi/common/ap/mutable_tensor.h"
#include "paddle/pir/include/dialect/pexpr/arithmetic_type_util.h"
#include "paddle/pir/include/dialect/pexpr/arithmetic_value.h"
#include "paddle/pir/include/dialect/pexpr/method_class.h"

namespace ap::kernel_dispatch {

using pexpr::ArithmeticType;
using pexpr::ArithmeticValue;
using pexpr::BuiltinBinaryFuncT;
using pexpr::BuiltinFuncType;
using pexpr::BuiltinUnaryFuncT;
using pexpr::CppArithmeticType;
using pexpr::CppPointerType;
using pexpr::Method;
using pexpr::MethodClass;
using pexpr::PointerType;
using pexpr::PointerValue;

namespace detail {

template <typename Val>
Result<Val> MutableTensorShapeGetAttr(const MutableTensor<Val>& tensor,
                                      const std::string&) {
  return tensor->dims;
}

template <typename T>
T* GetMutableTensorDataPtr(const pexpr::CppArithmeticType<T>&,
                           const MutableTensorData& tensor) {
  return tensor.template data<T>();
}

template <typename Val>
Result<Val> MutableTensorDataGetAttr(const MutableTensor<Val>& tensor,
                                     const std::string&) {
  phi::DataType dtype = tensor->tensor_data.dtype();
  const auto& arithmetic_type = pexpr::GetArithmeticTypeFromPhiDataType(dtype);
  ADT_RETURN_IF_ERROR(arithmetic_type);
  return arithmetic_type.GetOkValue().Match(
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

  static const char* Name() { return "MutableTensor"; }

  template <typename BuiltinUnarySymbol>
  static std::optional<BuiltinUnaryFuncT<ValueT>> GetBuiltinUnaryFunc() {
    return std::nullopt;
  }

  template <typename BultinBinarySymbol>
  static std::optional<BuiltinBinaryFuncT<ValueT>> GetBuiltinBinaryFunc() {
    if constexpr (std::is_same_v<BultinBinarySymbol,
                                 pexpr::builtin_symbol::GetAttr>) {
      return &Self::GetAttr;
    }
    return std::nullopt;
  }

  static adt::Result<ValueT> GetAttr(const ValueT& obj_val,
                                     const ValueT& attr_name_val) {
    const auto& opt_obj =
        MethodClass<ValueT>::template TryGet<MutableTensor<Val>>(obj_val);
    ADT_RETURN_IF_ERROR(opt_obj);
    const auto& obj = opt_obj.GetOkValue();
    const auto& opt_attr_name =
        MethodClass<ValueT>::template TryGet<std::string>(attr_name_val);
    ADT_RETURN_IF_ERROR(opt_attr_name);
    const auto& attr_name = opt_attr_name.GetOkValue();
    return detail::TensorGetAttr<Val>(obj, attr_name);
  }
};

}  // namespace ap::kernel_dispatch

namespace pexpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, ap::kernel_dispatch::MutableTensor<ValueT>> {
  using method_class = ap::kernel_dispatch::MutableTensorMethodClass<ValueT>;

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
