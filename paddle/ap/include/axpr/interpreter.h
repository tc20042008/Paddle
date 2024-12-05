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

#include "paddle/ap/include/axpr/value.h"

namespace ap::axpr {

class Interpreter {
 public:
  explicit Interpreter(const axpr::AttrMap<axpr::Value>& builtin_frame_attr_map)
      : builtin_frame_attr_map_(builtin_frame_attr_map) {}

  adt::Result<axpr::Value> Interpret(const Lambda<CoreExpr>& lambda,
                                     const std::vector<axpr::Value>& args);
  adt::Result<axpr::Value> Interpret(
      const Function<SerializableValue>& function,
      const std::vector<axpr::Value>& args);

  adt::Result<axpr::Value> InterpretModule(
      const Frame<SerializableValue>& const_global_frame,
      const Lambda<CoreExpr>& lambda);

 private:
  axpr::AttrMap<axpr::Value> builtin_frame_attr_map_;
};

}  // namespace ap::axpr
