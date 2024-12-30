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

#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/data_type_util.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/naive_class_ops.h"
#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/paddle/phi/scalar_helper.h"
#include "paddle/ap/include/paddle/pir/attr_adt_type_id.h"
#include "paddle/ap/include/paddle/pir/attribute.h"

namespace ap::paddle {

inline adt::Result<axpr::Value> PirAttributeToString(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 0);
  ADT_LET_CONST_REF(self, self_val.template CastTo<pir::Attribute>());
  std::ostringstream ss;
  ss << self;
  return ss.str();
}

inline axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetPirAttributeClass() {
  static auto cls(axpr::MakeBuiltinClass<axpr::Value>(
      "PirAttribute",
      [&](const auto& DoEach) { DoEach("__str__", &PirAttributeToString); }));
  return axpr::MakeGlobalNaiveClassOps<pir::Attribute>(cls);
}

template <typename T>
struct MakePirAttributeImpl;

template <>
struct MakePirAttributeImpl<pir::BoolAttribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(data_val, args.at(0).template CastTo<axpr::DataValue>());
    ADT_LET_CONST_REF(bool_val, data_val.template TryGet<bool>());
    pir::Attribute attr{
        pir::BoolAttribute::get(pir::IrContext::Instance(), bool_val)};
    return GetPirAttributeClass().New(attr);
  }
};

template <>
struct MakePirAttributeImpl<pir::Complex64Attribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(data_val, args.at(0).template CastTo<axpr::DataValue>());
    ADT_LET_CONST_REF(complex_val, data_val.template TryGet<axpr::complex64>());
    pir::Attribute attr{
        pir::Complex64Attribute::get(pir::IrContext::Instance(), complex_val)};
    return GetPirAttributeClass().New(attr);
  }
};

template <>
struct MakePirAttributeImpl<pir::Complex128Attribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(data_val, args.at(0).template CastTo<axpr::DataValue>());
    ADT_LET_CONST_REF(val, data_val.template TryGet<axpr::complex128>());
    pir::Attribute attr{
        pir::Complex128Attribute::get(pir::IrContext::Instance(), val)};
    return GetPirAttributeClass().New(attr);
  }
};

template <>
struct MakePirAttributeImpl<pir::FloatAttribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(data_val, args.at(0).template CastTo<axpr::DataValue>());
    ADT_LET_CONST_REF(val, data_val.template TryGet<float>());
    pir::Attribute attr{
        pir::FloatAttribute::get(pir::IrContext::Instance(), val)};
    return GetPirAttributeClass().New(attr);
  }
};

template <>
struct MakePirAttributeImpl<pir::DoubleAttribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(data_val, args.at(0).template CastTo<axpr::DataValue>());
    ADT_LET_CONST_REF(val, data_val.template TryGet<double>());
    pir::Attribute attr{
        pir::DoubleAttribute::get(pir::IrContext::Instance(), val)};
    return GetPirAttributeClass().New(attr);
  }
};

template <>
struct MakePirAttributeImpl<pir::Int32Attribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(data_val, args.at(0).template CastTo<axpr::DataValue>());
    ADT_LET_CONST_REF(val, data_val.template TryGet<int32_t>());
    pir::Attribute attr{
        pir::Int32Attribute::get(pir::IrContext::Instance(), val)};
    return GetPirAttributeClass().New(attr);
  }
};

template <>
struct MakePirAttributeImpl<pir::IndexAttribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(data_val, args.at(0).template CastTo<axpr::DataValue>());
    ADT_LET_CONST_REF(val, data_val.template TryGet<int64_t>());
    pir::Attribute attr{
        pir::IndexAttribute::get(pir::IrContext::Instance(), val)};
    return GetPirAttributeClass().New(attr);
  }
};

template <>
struct MakePirAttributeImpl<pir::Int64Attribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(data_val, args.at(0).template CastTo<axpr::DataValue>());
    ADT_LET_CONST_REF(val, data_val.template TryGet<int64_t>());
    pir::Attribute attr{
        pir::Int64Attribute::get(pir::IrContext::Instance(), val)};
    return GetPirAttributeClass().New(attr);
  }
};

template <>
struct MakePirAttributeImpl<pir::PointerAttribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(data_val,
                      args.at(0).template CastTo<axpr::PointerValue>());
    ADT_LET_CONST_REF(val, data_val.template TryGet<void*>());
    pir::Attribute attr{
        pir::PointerAttribute::get(pir::IrContext::Instance(), val)};
    return GetPirAttributeClass().New(attr);
  }
};

template <>
struct MakePirAttributeImpl<pir::TypeAttribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(type_val, args.at(0).template CastTo<pir::Type>());
    pir::Attribute attr{
        pir::TypeAttribute::get(pir::IrContext::Instance(), type_val)};
    return GetPirAttributeClass().New(attr);
  }
};

template <>
struct MakePirAttributeImpl<pir::StrAttribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(val, args.at(0).template CastTo<std::string>());
    pir::Attribute attr{
        pir::StrAttribute::get(pir::IrContext::Instance(), val)};
    return GetPirAttributeClass().New(attr);
  }
};

template <>
struct MakePirAttributeImpl<pir::ArrayAttribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    std::vector<pir::Attribute> attrs;
    attrs.reserve(args.size());
    for (const auto& arg : args) {
      ADT_LET_CONST_REF(elt, arg.template CastTo<pir::Attribute>());
      attrs.emplace_back(elt);
    }
    pir::Attribute attr{
        pir::ArrayAttribute::get(pir::IrContext::Instance(), attrs)};
    return GetPirAttributeClass().New(attr);
  }
};

template <>
struct MakePirAttributeImpl<pir::TensorNameAttribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(val, args.at(0).template CastTo<std::string>());
    pir::Attribute attr{
        pir::TensorNameAttribute::get(pir::IrContext::Instance(), val)};
    return GetPirAttributeClass().New(attr);
  }
};

template <>
struct MakePirAttributeImpl<pir::shape::SymbolAttribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    return adt::errors::NotImplementedError{
        std::string() + "pir." + pir::shape::SymbolAttribute::name() +
        "() not implemented"};
  }
};

template <>
struct MakePirAttributeImpl<::paddle::dialect::KernelAttribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    return adt::errors::NotImplementedError{
        std::string() + "pir." + ::paddle::dialect::KernelAttribute::name() +
        "() is not implemneted"};
  }
};

template <>
struct MakePirAttributeImpl<::paddle::dialect::IntArrayAttribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    std::vector<int64_t> int_array;
    int_array.reserve(args.size());
    for (const auto& arg : args) {
      ADT_LET_CONST_REF(elt, arg.template CastTo<int64_t>());
      int_array.emplace_back(elt);
    }
    pir::Attribute attr{::paddle::dialect::IntArrayAttribute::get(
        pir::IrContext::Instance(), int_array)};
    return GetPirAttributeClass().New(attr);
  }
};

inline adt::Result<phi::Scalar> ConvertDataValueToScalar(
    const axpr::DataValue& data_val) {
  return ScalarHelper{}.ConvertFromDataType(data_val);
}

template <>
struct MakePirAttributeImpl<::paddle::dialect::ScalarAttribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(data_val, args.at(0).template CastTo<axpr::DataValue>());
    ADT_LET_CONST_REF(val, ConvertDataValueToScalar(data_val));
    pir::Attribute attr{::paddle::dialect::ScalarAttribute::get(
        pir::IrContext::Instance(), val)};
    return GetPirAttributeClass().New(attr);
  }
};

template <>
struct MakePirAttributeImpl<::paddle::dialect::DataTypeAttribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(data_type, args.at(0).template CastTo<axpr::DataType>());
    ADT_LET_CONST_REF(phi_data_type,
                      axpr::GetPhiDataTypeFromDataType(data_type));
    pir::Attribute attr{::paddle::dialect::DataTypeAttribute::get(
        pir::IrContext::Instance(), phi_data_type)};
    return GetPirAttributeClass().New(attr);
  }
};

template <>
struct MakePirAttributeImpl<::paddle::dialect::PlaceAttribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(place, args.at(0).template CastTo<phi::Place>());
    pir::Attribute attr{::paddle::dialect::PlaceAttribute::get(
        pir::IrContext::Instance(), place)};
    return GetPirAttributeClass().New(attr);
  }
};

template <>
struct MakePirAttributeImpl<::paddle::dialect::DataLayoutAttribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(data_layout_str,
                      args.at(0).template CastTo<std::string>());
    std::optional<::common::DataLayout> data_layout;
    try {
      data_layout = ::common::StringToDataLayout(data_layout_str);
    } catch (const std::exception&) {
      return adt::errors::ValueError{"StringToDataLayout('" + data_layout_str +
                                     "') failed"};
    }
    pir::Attribute attr{::paddle::dialect::DataLayoutAttribute::get(
        pir::IrContext::Instance(), data_layout.value())};
    return GetPirAttributeClass().New(attr);
  }
};

template <>
struct MakePirAttributeImpl<::cinn::dialect::GroupInfoAttribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    return adt::errors::NotImplementedError{
        std::string() + "pir." + ::cinn::dialect::GroupInfoAttribute::name() +
        "() is not implemneted"};
  }
};

template <>
struct MakePirAttributeImpl<::cinn::dialect::CINNKernelInfoAttribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    return adt::errors::NotImplementedError{
        std::string() + "pir." +
        ::cinn::dialect::CINNKernelInfoAttribute::name() +
        "() is not implemneted"};
  }
};

template <>
struct MakePirAttributeImpl<UnclassifiedAttribute> {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    return adt::errors::NotImplementedError{std::string() + "pir." +
                                            UnclassifiedAttribute::name() +
                                            "() is not implemneted"};
  }
};

}  // namespace ap::paddle
