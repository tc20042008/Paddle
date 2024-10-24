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

#include "ap/axpr/data_type_util.h"
#include "ap/axpr/method_class.h"
#include "ap/paddle/ddim.h"

namespace ap::paddle {

template <typename ValueT>
struct DDimMethodClass {
  using This = DDimMethodClass;
  using Self = DDim;

  adt::Result<ValueT> ToString(const Self& self) {
    std::ostringstream ss;
    ss << "[";
    for (int i = 0; i < self.size(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << self.at(i);
    }
    ss << "]";
    return ss.str();
  }

  adt::Result<ValueT> GetItem(const Self& self, const ValueT& index_val) {
    ADT_LET_CONST_REF(index, index_val.template TryGet<int64_t>())
        << adt::errors::TypeError{std::string() +
                                  "'DDim.__get_item__' takes integers, not " +
                                  axpr::GetTypeName(index_val) + "."};
    ADT_CHECK(index < self.size())
        << adt::errors::IndexError{"list index out of range"};
    return self.at(index);
  }
};

}  // namespace ap::paddle

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, paddle::DDim>
    : public paddle::DDimMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<paddle::DDim>> {};

}  // namespace ap::axpr
