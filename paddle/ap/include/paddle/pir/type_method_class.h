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

#include "paddle/ap/include/paddle/pir/type.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/dialect/shape/ir/shape_attribute.h"

namespace ap::paddle {

inline adt::Result<axpr::Value> PirTypeString(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_LET_CONST_REF(self, self_val.template CastTo<pir::Type>());
  std::ostringstream ss;
  ss << self;
  return ss.str();
}

inline axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetPirTypeClass() {
  static auto cls(axpr::MakeBuiltinClass<axpr::Value>(
      "PirType",
      [&](const auto& DoEach) { DoEach("__str__", &PirTypeString); }));
  return axpr::MakeGlobalNaiveClassOps<pir::Type>(cls);
}

template <typename T>
struct MakePirTypeImpl;

template <>
struct MakePirTypeImpl<NullType> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    pir::Type type;
    return GetPirTypeClass().New(type);
  }
};

template <>
struct MakePirTypeImpl<::pir::VectorType> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    std::vector<pir::Type> types;
    for (const auto& arg : args) {
      ADT_LET_CONST_REF(elt, arg.template CastTo<pir::Type>());
      types.emplace_back(elt);
    }
    const pir::Type type{
        pir::VectorType::get(pir::IrContext::Instance(), types)};
    return GetPirTypeClass().New(type);
  }
};

template <>
struct MakePirTypeImpl<::pir::DenseTensorType> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 3);
    ADT_LET_CONST_REF(type, args.at(0).template CastTo<pir::Type>());
    ADT_LET_CONST_REF(int_list,
                      args.at(1).template CastTo<adt::List<axpr::Value>>());
    std::vector<int64_t> dims;
    dims.reserve(int_list->size());
    for (const auto& int_val : *int_list) {
      ADT_LET_CONST_REF(elt, int_val.template CastTo<int64_t>());
      dims.emplace_back(elt);
    }
    ::common::DDim ddim(dims.data(), dims.size());
    ADT_LET_CONST_REF(data_layout_str,
                      args.at(2).template CastTo<std::string>());
    std::optional<::common::DataLayout> data_layout;
    try {
      data_layout = ::common::StringToDataLayout(data_layout_str);
    } catch (const std::exception&) {
      return adt::errors::ValueError{"StringToDataLayout('" + data_layout_str +
                                     "') failed"};
    }
    ADT_CHECK(data_layout.has_value());
    const pir::Type dense_tensor_type{pir::DenseTensorType::get(
        pir::IrContext::Instance(), type, ddim, data_layout.value())};
    return GetPirTypeClass().New(dense_tensor_type);
  }
};

template <>
struct MakePirTypeImpl<::pir::BFloat16Type> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    const pir::Type type{pir::BFloat16Type::get(pir::IrContext::Instance())};
    return GetPirTypeClass().New(type);
  }
};

template <>
struct MakePirTypeImpl<::pir::Float16Type> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    const pir::Type type{pir::Float16Type::get(pir::IrContext::Instance())};
    return GetPirTypeClass().New(type);
  }
};

template <>
struct MakePirTypeImpl<::pir::Float32Type> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    const pir::Type type{pir::Float32Type::get(pir::IrContext::Instance())};
    return GetPirTypeClass().New(type);
  }
};

template <>
struct MakePirTypeImpl<::pir::Float64Type> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    const pir::Type type{pir::Float64Type::get(pir::IrContext::Instance())};
    return GetPirTypeClass().New(type);
  }
};

template <>
struct MakePirTypeImpl<::pir::Int8Type> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    const pir::Type type{pir::Int8Type::get(pir::IrContext::Instance())};
    return GetPirTypeClass().New(type);
  }
};

template <>
struct MakePirTypeImpl<::pir::UInt8Type> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    const pir::Type type{pir::UInt8Type::get(pir::IrContext::Instance())};
    return GetPirTypeClass().New(type);
  }
};

template <>
struct MakePirTypeImpl<::pir::Int16Type> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    const pir::Type type{pir::Int16Type::get(pir::IrContext::Instance())};
    return GetPirTypeClass().New(type);
  }
};

template <>
struct MakePirTypeImpl<::pir::Int32Type> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    const pir::Type type{pir::Int32Type::get(pir::IrContext::Instance())};
    return GetPirTypeClass().New(type);
  }
};

template <>
struct MakePirTypeImpl<::pir::Int64Type> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    const pir::Type type{pir::Int64Type::get(pir::IrContext::Instance())};
    return GetPirTypeClass().New(type);
  }
};

template <>
struct MakePirTypeImpl<::pir::IndexType> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    const pir::Type type{pir::IndexType::get(pir::IrContext::Instance())};
    return GetPirTypeClass().New(type);
  }
};

template <>
struct MakePirTypeImpl<::pir::BoolType> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    const pir::Type type{pir::BoolType::get(pir::IrContext::Instance())};
    return GetPirTypeClass().New(type);
  }
};

template <>
struct MakePirTypeImpl<::pir::Complex64Type> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    const pir::Type type{pir::Complex64Type::get(pir::IrContext::Instance())};
    return GetPirTypeClass().New(type);
  }
};

template <>
struct MakePirTypeImpl<::pir::Complex128Type> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    const pir::Type type{pir::Complex128Type::get(pir::IrContext::Instance())};
    return GetPirTypeClass().New(type);
  }
};

template <>
struct MakePirTypeImpl<::paddle::dialect::SelectedRowsType> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    const pir::Type type{
        ::paddle::dialect::SelectedRowsType::get(pir::IrContext::Instance())};
    return GetPirTypeClass().New(type);
  }
};

template <>
struct MakePirTypeImpl<::paddle::dialect::DenseTensorArrayType> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 3);
    ADT_LET_CONST_REF(type, args.at(0).template CastTo<pir::Type>());
    ADT_LET_CONST_REF(int_list,
                      args.at(1).template CastTo<adt::List<axpr::Value>>());
    std::vector<int64_t> dims;
    dims.reserve(int_list->size());
    for (const auto& int_val : *int_list) {
      ADT_LET_CONST_REF(elt, int_val.template CastTo<int64_t>());
      dims.emplace_back(elt);
    }
    ::common::DDim ddim(dims.data(), dims.size());
    ADT_LET_CONST_REF(data_layout_str,
                      args.at(2).template CastTo<std::string>());
    std::optional<::common::DataLayout> data_layout;
    try {
      data_layout = ::common::StringToDataLayout(data_layout_str);
    } catch (const std::exception&) {
      return adt::errors::ValueError{"StringToDataLayout('" + data_layout_str +
                                     "') failed"};
    }
    ADT_CHECK(data_layout.has_value());
    const pir::Type dense_tensor_type{
        ::paddle::dialect::DenseTensorArrayType::get(
            pir::IrContext::Instance(), type, ddim, data_layout.value())};
    return GetPirTypeClass().New(dense_tensor_type);
  }
};

template <>
struct MakePirTypeImpl<::paddle::dialect::SparseCooTensorType> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    return adt::errors::NotImplementedError{
        std::string() + ::paddle::dialect::SparseCooTensorType::name() +
        "() is not implemented"};
  }
};

template <>
struct MakePirTypeImpl<::paddle::dialect::SparseCsrTensorType> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    return adt::errors::NotImplementedError{
        std::string() + ::paddle::dialect::SparseCsrTensorType::name() +
        "() is not implemented"};
  }
};

template <>
struct MakePirTypeImpl<UnclassifiedType> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    return adt::errors::NotImplementedError{
        std::string() + UnclassifiedType::name() + "() is not implemented"};
  }
};

}  // namespace ap::paddle
