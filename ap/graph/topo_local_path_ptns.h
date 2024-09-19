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
#include "ap/adt/adt.h"
#include "ap/graph/jump.h"
#include "ap/graph/path_ptn.h"
#include "ap/graph/size_to_features.h"

namespace ap::graph {

struct TopoLocalPathPtnsImpl {
  SizeToFeatures<PathPtn<UpJump>> size_to_up_paths;
  SizeToFeatures<PathPtn<DownJump>> size_to_down_paths;
};

DEFINE_ADT_RC(TopoLocalPathPtns, TopoLocalPathPtnsImpl);

}  // namespace ap::graph
