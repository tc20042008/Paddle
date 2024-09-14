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

#include "ap/axpr/adt.h"
#include "ap/axpr/constants.h"
#include "ap/axpr/method_class.h"
#include "ap/axpr/starred.h"

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, adt::List<ValueT>> {
  using Self = adt::List<ValueT>;

  adt::Result<ValueT> GetItem(const Self& self, const ValueT& idx) {
    return idx.Match(
        [&](int64_t index) -> Result<ValueT> {
          if (index < 0) {
            index += self->size();
          }
          if (index >= 0 && index < self->size()) {
            return self->at(index);
          }
          return adt::errors::IndexError{"list index out of range"};
        },
        [&](const auto&) -> Result<ValueT> {
          return adt::errors::TypeError{std::string() +
                                        "list indices must be integers, not " +
                                        MethodClass<ValueT>::Name(idx)};
        });
  }

  adt::Result<ValueT> Starred(const Self& self) {
    return ap::axpr::Starred<ValueT>{self};
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<adt::List<ValueT>>> {
  using Self = TypeImpl<adt::List<ValueT>>;

  using This = MethodClassImpl<ValueT, Self>;
};

}  // namespace ap::axpr
