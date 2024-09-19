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

#include <map>
#include "ap/graph/adt.h"
#include "ap/graph/jump.h"
#include "ap/graph/path_ptn.h"

namespace ap::graph {

template <typename FeatureT>
struct SizeToFeaturesImpl {
  using NumberOfIdentityFeatures = std::size_t;
  using Int2Features =
      std::vector<std::map<FeatureT, NumberOfIdentityFeatures>>;

  Int2Features size2features;

  bool Includes(const SizeToFeaturesImpl& other) const {
    if (this->size2features.size() < other.size2features.size()) {
      return false;
    }
    for (int i = 0; i < other.size2features.size(); ++i) {
      const auto& this_map = this->size2features.at(i);
      const auto& other_map = other.size2features.at(i);
      if (this_map.size() < other_map.size()) {
        return false;
      }
      for (const auto& [other_feature, other_feature_size] : other_map) {
        const auto& this_iter = this_map.find(other_feature);
        if (this_iter == this_map.end()) {
          return false;
        }
        const auto& this_feature_size = this_iter->second;
        if (this_feature_size < other_feature_size) {
          return false;
        }
      }
    }
    return true;
  }
};

template <typename FeatureT>
DEFINE_ADT_RC(SizeToFeatures, SizeToFeaturesImpl<FeatureT>);

}  // namespace ap::graph
