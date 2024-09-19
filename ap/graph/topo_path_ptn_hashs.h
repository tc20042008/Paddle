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

#include <set>
#include "ap/adt/adt.h"
#include "ap/graph/distance_to_features.h"
#include "ap/graph/hash_value_type.h"
#include "ap/graph/topo_local_path_ptns.h"

namespace ap::graph {

struct TopoPathPtnHashs;

struct TopoPathPtnHashsImpl {
  Distance2Features<HashValueType<TopoLocalPathPtns>>
      distance2upstream_hash_values;
  Distance2Features<HashValueType<TopoLocalPathPtns>>
      distance2downstream_hash_values;

  bool Includes(const TopoPathPtnHashsImpl& other) const {
    return distance2upstream_hash_values->Includes(
               *other.distance2upstream_hash_values) &&
           distance2downstream_hash_values->Includes(
               *other.distance2downstream_hash_values);
  }
};

DEFINE_ADT_RC(TopoPathPtnHashs, TopoPathPtnHashsImpl);

}  // namespace ap::graph
