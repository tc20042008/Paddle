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

#include "ap/axpr/method_class.h"
#include "ap/axpr/type.h"
#include "ap/drr/drr_value.h"
#include "ap/drr/tags.h"
#include "ap/drr/unbound_packed_ir_value.h"

namespace ap::drr {

struct UnboundPackedIrValueMethodClass {
  using Self = drr::UnboundPackedIrValue<drr::Node>;

  static adt::Result<axpr::Value> ToString(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const void* ptr = self.__adt_rc_shared_ptr_raw_ptr();
    std::ostringstream ss;
    ss << "<" << drr::Type<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  static adt::Result<axpr::Value> Hash(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const void* ptr = self.__adt_rc_shared_ptr_raw_ptr();
    return reinterpret_cast<int64_t>(ptr);
  }
};

inline const axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>&
GetUnboundPackedIrValueClass() {
  using ClassT = axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>;
  using Impl = UnboundPackedIrValueMethodClass;
  using TT = drr::Type<UnboundPackedIrValue<drr::Node>>;
  static ClassT cls(
      axpr::MakeBuiltinClass<axpr::Value>(TT{}.Name(), [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
      }));
  return cls;
}

}  // namespace ap::drr
