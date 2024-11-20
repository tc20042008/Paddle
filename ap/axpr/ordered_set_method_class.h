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

#include "ap/axpr/builtin_func_type.h"
#include "ap/axpr/method_class.h"
#include "ap/axpr/ordered_dict.h"
#include "ap/axpr/to_string.h"

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, OrderedSet<ValueT>> {
  using Self = OrderedSet<ValueT>;

  using This = MethodClassImpl<ValueT, Self>;

  adt::Result<ValueT> Length(const Self& self) {
    return static_cast<int64_t>(self->items().size());
  }

  adt::Result<ValueT> ToString(const Self& self) {
    std::ostringstream ss;
    ss << "OrderedSet([";
    int i = 0;
    for (const auto& elt : self->items()) {
      if (i++ > 0) {
        ss << ", ";
      }
      ADT_LET_CONST_REF(elt_str, axpr::ToString(elt));
      ss << elt_str;
    }
    ss << "])";
    return ss.str();
  }

  adt::Result<ValueT> Starred(const Self& self) {
    adt::List<ValueT> lst;
    lst->reserve(self->items().size());
    for (const auto& elt : self->items()) {
      lst->emplace_back(elt);
    }
    return ap::axpr::Starred<ValueT>{lst};
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<OrderedSet<ValueT>>> {
  using Val = ValueT;
  using Self = TypeImpl<OrderedSet<ValueT>>;
  using This = MethodClassImpl<ValueT, Self>;

  adt::Result<ValueT> Call(const Self& self) {
    return axpr::Method<ValueT>{self, &This::Construct};
  }

  static adt::Result<ValueT> Construct(const ValueT&,
                                       const std::vector<ValueT>& args) {
    if (args.size() == 0) {
      return OrderedSet<ValueT>{};
    }
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "OrderedSet() takes 1 argument but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(lst, args.at(0).template TryGet<adt::List<ValueT>>())
        << adt::errors::TypeError{
               std::string() +
               "the argument 1 of OrderedSet() should be list, " +
               axpr::GetTypeName(args.at(0)) + " found."};
    OrderedSet<ValueT> ordered_dict{};
    for (const auto& elt : *lst) {
      ADT_RETURN_IF_ERR(ordered_dict->Insert(elt));
    }
    return ordered_dict;
  }
};

}  // namespace ap::axpr
