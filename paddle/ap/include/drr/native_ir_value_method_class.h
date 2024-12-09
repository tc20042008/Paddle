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

#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/naive_class_ops.h"
#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/drr/drr_value.h"
#include "paddle/ap/include/drr/native_ir_value.h"
#include "paddle/ap/include/drr/tags.h"

namespace ap::drr {

struct SrcPtnNativeIrValueMethodClassImpl {
  using Self = drr::tSrcPtn<drr::NativeIrValue<drr::Node>>;
  using This = SrcPtnNativeIrValueMethodClassImpl;

  static adt::Result<axpr::Value> ToString(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    std::ostringstream ss;
    const void* ptr = self.value().__adt_rc_shared_ptr_raw_ptr();
    ss << "<" << drr::Type<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  static adt::Result<axpr::Value> Hash(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const void* ptr = self.value().__adt_rc_shared_ptr_raw_ptr();
    return reinterpret_cast<int64_t>(ptr);
  }

  static adt::Result<axpr::Value> Starred(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return adt::errors::TypeError{
        std::string() +
        "Only SrcPtnPackedIrValue and ResPtnPackedIrValue tensors can be "
        "unpacked. tensor '" +
        self.value()->name + "' is of type 'SrcPtnNativeIrValue'"};
  }
};

inline axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetSrcPtnNativeIrValueClass() {
  using Impl = SrcPtnNativeIrValueMethodClassImpl;
  using TT = drr::Type<drr::tSrcPtn<drr::NativeIrValue<drr::Node>>>;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>(TT{}.Name(), [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
        Define("__starred__", &Impl::Starred);
      }));
  using Self = typename Impl::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

struct ResPtnNativeIrValueMethodClass {
  using Self = drr::tResPtn<drr::NativeIrValue<drr::Node>>;
  using This = ResPtnNativeIrValueMethodClass;

  static adt::Result<axpr::Value> ToString(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    std::ostringstream ss;
    const void* ptr = self.value().__adt_rc_shared_ptr_raw_ptr();
    ss << "<" << drr::Type<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  static adt::Result<axpr::Value> Hash(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const void* ptr = self.value().__adt_rc_shared_ptr_raw_ptr();
    return reinterpret_cast<int64_t>(ptr);
  }

  static adt::Result<axpr::Value> Starred(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return adt::errors::TypeError{
        std::string() +
        "Only SrcPtnPackedIrValue and ResPtnPackedIrValue tensors can be "
        "unpacked. tensor '" +
        self.value()->name + "' is of type 'ResPtnNativeIrValue'"};
  }
};

inline axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetResPtnNativeIrValueClass() {
  using Impl = ResPtnNativeIrValueMethodClass;
  using TT = drr::Type<drr::tResPtn<drr::NativeIrValue<drr::Node>>>;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>(TT{}.Name(), [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
        Define("__starred__", &Impl::Starred);
      }));
  using Self = typename Impl::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::drr
