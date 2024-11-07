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

#include <list>
#include <unordered_map>
#include "ap/kernel_define/kernel_arg_id.h"

namespace ap::kernel_define {

template <typename ValueT, typename IrNodeT>
struct KernelArgIdOrderedMapImpl {
  using Pair = std::pair<KernelArgId<IrNodeT>, ValueT>;
  using Pairs = std::list<Pair>;
  Pairs pairs;
  std::unordered_map<KernelArgId<IrNodeT>, Pairs::iterator> kernel_arg_id2iter;

  adt::Result<adt::Ok> Insert(const KernelArgId<IrNodeT>& kernel_arg_id,
                              const ValueT& val) {
    const auto& pair = std::make_pair(kernel_arg_id, val);
    const auto map_iter = kernel_arg_id2iter.find(kernel_arg_id);
    if (map_iter != kernel_arg_id2iter.end()) {
      *map_iter->second = pair;
    } else {
      const auto pair_iter = pairs.insert(pairs.end(), pair);
      ADT_CHECK(kernel_arg_id2iter->emplace(kernel_arg_id, pair_iter).second);
    }
    return adt::Ok{};
  }
};

template <typename ValueT, typename IrNodeT>
DEFINE_ADT_RC(KernelArgIdOrderedMap,
              KernelArgIdOrderedMapImpl<ValueT, IrNodeT>);

}  // namespace ap::kernel_define
