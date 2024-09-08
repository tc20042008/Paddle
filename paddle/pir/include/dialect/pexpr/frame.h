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

#include <string>
#include "paddle/pir/include/dialect/pexpr/adt.h"
#include "paddle/pir/include/dialect/pexpr/object.h"

namespace pexpr {

template <typename ValueT>
struct Frame {
  Result<ValueT> Get(const std::string& var) const {
    return frame_obj->Get(var);
  }

  bool Set(const std::string& var, const ValueT& val) {
    return frame_obj->Set(var, val);
  }

  bool HasVar(const std::string& var) const {
    return frame_obj->find(var) != frame_obj->end();
  }

  void ClearFrame() { frame_obj->clear(); }

  Object<ValueT> frame_obj;
};

}  // namespace pexpr
