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
#include "ap/axpr/attr_map.h"
#include "ap/axpr/builtin_frame_util.h"

namespace ap::index_expr {

template <typename ValueT>
AttrMap<ValueT> MakeBuiltinFrameAttrMap() {
  AttrMap<ValueT> attr_map;
  auto Insert = [&](const std::string& k, const ValueT& v) {
    attr_map->Set(k, v);
  };
  VisitEachBuiltinFrameAttr<ValueT>(Insert);
  return attr_map;
}

}  // namespace ap::index_expr
