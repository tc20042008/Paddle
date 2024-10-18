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
#include "ap/axpr/object.h"
#include "ap/axpr/type.h"

namespace ap::axpr {

template <typename ValueT>
struct PackedArgsImpl {
  adt::List<ValueT> args;
  axpr::Object<ValueT> kwargs;

  bool operator==(const PackedArgsImpl& other) const {
    return this->args == other.args && this->kwargs == other.kwargs;
  }
};

template <typename ValueT>
DEFINE_ADT_RC(PackedArgs, PackedArgsImpl<ValueT>);

template <typename ValueT>
PackedArgs<ValueT> CastToPackedArgs(
    const std::vector<ValueT>& packed_args_vec) {
  if (packed_args_vec.size() == 1 &&
      packed_args_vec.at(0).template Has<PackedArgs<ValueT>>()) {
    return packed_args_vec.at(0).template Get<PackedArgs<ValueT>>();
  } else {
    adt::List<ValueT> pos_args{};
    pos_args->reserve(packed_args_vec.size());
    pos_args->assign(packed_args_vec.begin(), packed_args_vec.end());
    return PackedArgs<ValueT>{pos_args, Object<ValueT>{}};
  }
}

template <typename ValueT>
struct TypeImpl<PackedArgs<ValueT>> : public std::monostate {
  using value_type = PackedArgs<ValueT>;

  const char* Name() const { return "__builtin_PackedArgs__"; }
};

}  // namespace ap::axpr
