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

#include "ap/include/axpr/hash.h"
#include "ap/include/axpr/ordered_dict.h"

namespace ap::axpr {

template <typename ValueT>
struct OrderedDictHelper {
  adt::Result<adt::Ok> Insert(OrderedDictImpl<ValueT>* dict,
                              const ValueT& key,
                              const ValueT& val) {
    ADT_LET_CONST_REF(hash_value, axpr::Hash(key));
    auto* lst = &dict->hash_value2pair_iters[hash_value];
    for (auto iter : *lst) {
      if (iter->first == key) {
        iter->second = val;
        return adt::Ok{};
      }
    }
    lst->emplace_back(
        dict->items.insert(dict->items.end(), std::pair{key, value}));
    return adt::Ok{};
  }
};

}  // namespace ap::axpr
