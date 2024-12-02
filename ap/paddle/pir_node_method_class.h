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

#include "ap/axpr/dim_expr_method_class.h"
#include "ap/paddle/pir_node.h"

namespace ap::paddle {

template <typename ValueT>
struct NativeIrValueMethodClass {
  using This = NativeIrValueMethodClass;
  using Self = NativeIrValue;

  adt::Result<ValueT> ToString(const Self& self) {
    std::ostringstream ss;
    const auto* ptr = self.value.impl();
    ss << "<" << axpr::TypeImpl<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  adt::Result<ValueT> Hash(const Self& self) {
    return static_cast<int64_t>(std::hash<Self>()(self));
  }

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    if (attr_name == "dtype") {
      return GetDataType(self);
    } else if (attr_name == "shape") {
      return GetShape(self);
    }
    return adt::errors::TypeError{std::string() +
                                  "NativeIrValue instance has no attribute '" +
                                  attr_name + "'."};
  }

  adt::Result<ValueT> GetShape(const Self& self) {
    ADT_LET_CONST_REF(shape_ptr, self.GetShapeDimExprsPtr());
    adt::List<ValueT> lst;
    lst->reserve(shape_ptr->size());
    for (const auto& dim_expr : *shape_ptr) {
      axpr::BuiltinClassInstance<ValueT> instance{
          axpr::GetDimExprClass<ValueT>(), dim_expr};
      lst->emplace_back(instance);
    }
    return lst;
  }

  adt::Result<ValueT> GetDataType(const Self& self) {
    ADT_LET_CONST_REF(dtype, self.GetDataType());
    return dtype;
  }
};

template <typename ValueT>
struct PackedIrValueMethodClass {
  using This = PackedIrValueMethodClass;
  using Self = PackedIrValue;

  adt::Result<ValueT> ToString(const Self& self) {
    std::ostringstream ss;
    const pir::Operation* ptr = self.fusion_op;
    ss << "<" << axpr::TypeImpl<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  adt::Result<ValueT> Hash(const Self& self) {
    const pir::Operation* ptr = self.fusion_op;
    return reinterpret_cast<int64_t>(ptr);
  }
};

template <typename ValueT>
struct RefIrValueMethodClass {
  using This = RefIrValueMethodClass;
  using Self = RefIrValue;

  adt::Result<ValueT> ToString(const Self& self) {
    std::ostringstream ss;
    const auto* ptr = self.ref_node_info.__adt_rc_shared_ptr_raw_ptr();
    ss << "<" << axpr::TypeImpl<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  adt::Result<ValueT> Hash(Self self) {
    return reinterpret_cast<int64_t>(
        self.ref_node_info.__adt_rc_shared_ptr_raw_ptr());
  }

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    if (attr_name == "dtype") {
      return GetDataType(self);
    } else if (attr_name == "shape") {
      return GetShape(self);
    }
    return adt::errors::TypeError{std::string() +
                                  "NativeIrValue instance has no attribute '" +
                                  attr_name + "'."};
  }

  adt::Result<ValueT> GetShape(const Self& self) {
    ADT_LET_CONST_REF(ir_value, self.GetOwnerNativeIrValue());
    ADT_LET_CONST_REF(shape_ptr, ir_value.GetShapeDimExprsPtr());
    adt::List<ValueT> lst;
    lst->reserve(shape_ptr->size());
    for (const auto& dim_expr : *shape_ptr) {
      axpr::BuiltinClassInstance<ValueT> instance{
          axpr::GetDimExprClass<ValueT>(), dim_expr};
      lst->emplace_back(instance);
    }
    return lst;
  }

  adt::Result<ValueT> GetDataType(const Self& self) {
    ADT_LET_CONST_REF(ir_value, self.GetOwnerNativeIrValue());
    ADT_LET_CONST_REF(dtype, ir_value.GetDataType());
    return dtype;
  }
};

template <typename ValueT>
struct NativeIrOpMethodClass {
  using This = NativeIrOpMethodClass;
  using Self = NativeIrOp;

  adt::Result<ValueT> ToString(const Self& self) {
    std::ostringstream ss;
    const auto* ptr = self.op;
    ss << "<" << axpr::TypeImpl<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  adt::Result<ValueT> Hash(Self self) {
    const pir::Operation* ptr = self.op;
    return reinterpret_cast<int64_t>(ptr);
  }
};

template <typename ValueT>
struct PackedIrOpMethodClass {
  using This = PackedIrOpMethodClass;
  using Self = PackedIrOp;

  adt::Result<ValueT> ToString(const Self& self) {
    std::ostringstream ss;
    const pir::Operation* ptr = self.fusion_op;
    ss << "<" << axpr::TypeImpl<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  adt::Result<ValueT> Hash(Self self) {
    const pir::Operation* ptr = self.fusion_op;
    return reinterpret_cast<int64_t>(ptr);
  }
};

template <typename ValueT>
struct RefIrOpMethodClass {
  using This = RefIrOpMethodClass;
  using Self = RefIrOp;

  adt::Result<ValueT> ToString(const Self& self) {
    std::ostringstream ss;
    const auto* ptr = self.ref_node_info.__adt_rc_shared_ptr_raw_ptr();
    ss << "<" << axpr::TypeImpl<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  adt::Result<ValueT> Hash(Self self) {
    return reinterpret_cast<int64_t>(
        self.ref_node_info.__adt_rc_shared_ptr_raw_ptr());
  }
};

}  // namespace ap::paddle

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, ap::paddle::NativeIrValue>
    : public paddle::NativeIrValueMethodClass<ValueT> {};
template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<ap::paddle::NativeIrValue>> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, ap::paddle::PackedIrValue>
    : public paddle::PackedIrValueMethodClass<ValueT> {};
template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<ap::paddle::PackedIrValue>> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, ap::paddle::RefIrValue>
    : public paddle::RefIrValueMethodClass<ValueT> {};
template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<ap::paddle::RefIrValue>> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, ap::paddle::NativeIrOp>
    : public paddle::NativeIrOpMethodClass<ValueT> {};
template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<ap::paddle::NativeIrOp>> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, ap::paddle::PackedIrOp>
    : public paddle::PackedIrOpMethodClass<ValueT> {};
template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<ap::paddle::PackedIrOp>> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, ap::paddle::RefIrOp>
    : public paddle::RefIrOpMethodClass<ValueT> {};
template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<ap::paddle::RefIrOp>> {};

}  // namespace ap::axpr
