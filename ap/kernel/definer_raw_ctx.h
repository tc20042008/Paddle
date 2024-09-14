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
#include "ap/axpr/type.h"
#include "ap/kernel/adt.h"
#include "ap/kernel/data_type.h"
#include "paddle/cinn/adt/adt.h"

namespace ap::kernel_define {

namespace adt = ::cinn::adt;

struct DefinerRawCtxImpl {
  bool operator==(const DefinerRawCtxImpl& other) { return &other == this; }
};
DEFINE_ADT_RC(DefinerRawCtx, DefinerRawCtxImpl);

}  // namespace ap::kernel_define

namespace ap::axpr {

template <>
struct TypeImpl<ap::kernel_define::DefinerRawCtx> : public std::monostate {
  using value_type = ap::kernel_define::DefinerRawCtx;

  const char* Name() const { return "DefinerRawCtx"; }
};

}  // namespace ap::axpr
