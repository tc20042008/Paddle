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
#include "ap/code_gen/kernel_arg_id.h"

namespace ap::code_gen {

template <typename BirNode>
struct KernelArgIdOrderedSetImpl {
  using KernelArgIds = std::list<KernelArgId<BirNode>>;
  KernelArgIds kernel_arg_ids;
  std::unordered_map<KernelArgId<BirNode>, typename KernelArgIds::iterator>
      kernel_arg_id2iter;

  adt::Result<bool> Emplace(const KernelArgId<BirNode>& kernel_arg_id) {
    if (kernel_arg_id2iter.count(kernel_arg_id) > 0) {
      return false;
    }
    const auto iter =
        kernel_arg_ids.insert(kernel_arg_ids.end(), kernel_arg_id);
    ADT_CHECK(kernel_arg_id2iter.emplace(kernel_arg_id, iter).second);
    return true;
  }
};

template <typename BirNode>
DEFINE_ADT_RC(KernelArgIdOrderedSet, KernelArgIdOrderedSetImpl<BirNode>);

}  // namespace ap::code_gen
