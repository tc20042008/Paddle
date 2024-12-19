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
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/paddle/phi/place_method_class.h"
#include "paddle/ap/include/paddle/pir/attribute_method_class.h"
#include "paddle/ap/include/paddle/pir/pir.h"
#include "paddle/ap/include/paddle/pir/type_method_class.h"

namespace ap::paddle {

inline axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetPirClass() {
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>("pir", [&](const auto& DoEach) {
        DoEach("CPUPlace", &CreateCPUPlace);
        DoEach("GPUPlace", &CreateGPUPlace);
        DoEach("GPUPinnedPlace", &CreateGPUPinnedPlace);
        DoEach("XPUPlace", &CreateXPUPlace);
        DoEach("IPUPlace", &CreateIPUPlace);
        DoEach("CustomPlace", &CreateCustomPlace);
#define YIELD_MAKE_ATTRIBUTE(attr_type) \
  DoEach(attr_type::name(), &MakePirAttributeImpl<attr_type>::Call);
        FOR_EACH_PIR_ATTRIBUTE_TYPE(YIELD_MAKE_ATTRIBUTE);
#undef YIELD_MAKE_ATTRIBUTE

#define YIELD_MAKE_TYPE(cls) DoEach(cls::name(), &MakePirTypeImpl<cls>::Call);
        FOR_EACH_PIR_ALTERNATIVE_TYPLE(YIELD_MAKE_TYPE);
#undef YIELD_MAKE_TYPE
      }));
  return axpr::MakeGlobalNaiveClassOps<Pir>(cls);
}

}  // namespace ap::paddle
