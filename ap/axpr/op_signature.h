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
#include "ap/axpr/index_expr.h"

namespace ap::axpr {

template <typename T>
struct InputSignature {
  adt::List<T> descriptors;

  bool operator==(const InputSignature& other) const {
    return other.descriptors == this->descriptors;
  }
};

template <typename T>
struct OutputSignature {
  adt::List<T> descriptors;

  bool operator==(const OutputSignature& other) const {
    return other.descriptors == this->descriptors;
  }
};

template <typename T>
struct OpSignature {
  InputSignature<T> in_signature;
  OutputSignature<T> out_signature;

  bool operator==(const OpSignature& other) const {
    return other.in_signature == this->in_signature &&
           other.out_signature == this->out_signature;
  }
};

}  // namespace ap::axpr
