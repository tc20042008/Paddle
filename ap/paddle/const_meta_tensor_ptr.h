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

#include "ap/adt/adt.h"
#include "ap/axpr/const_std_vector_ptr.h"
#include "ap/axpr/type.h"
#include "paddle/phi/core/meta_tensor.h"

namespace ap::paddle {

using ConstMetaTensorPtr = const ::phi::MetaTensor*;

}

namespace ap::axpr {

template <>
struct TypeImpl<paddle::ConstMetaTensorPtr> : public std::monostate {
  using std::monostate::monostate;

  const char* Name() const { return "ConstMetaTensorPtr"; }
};

template <>
struct GetConstVectorPtrNameHelper<paddle::ConstMetaTensorPtr> {
  static const char* Call() {
    return "const_std_vector_ConstMetaTensorPtr_ptr";
  }
};

}  // namespace ap::axpr
