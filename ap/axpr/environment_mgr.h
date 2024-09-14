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
#include <string>
#include "ap/axpr/adt.h"
#include "ap/axpr/environment.h"
#include "ap/axpr/object.h"

namespace ap::axpr {

// Watch out circle references:
// Value   ->   Closure
//   /\              |
//   |               V
//  Frame  <-  Environment
// To avoid memory leak, make sure every Environment instances are created by
// EnvironmentManager;
template <typename ValueT>
class EnvironmentManager {
 public:
  EnvironmentManager() {}
  ~EnvironmentManager() { ClearAllFrames(); }

  std::shared_ptr<Environment<ValueT>> New(
      const std::shared_ptr<Environment<ValueT>>& env) {
    auto ptr = Environment<ValueT>::New(env);
    weak_envs_.push_back(ptr);
    return ptr;
  }

  std::shared_ptr<Environment<ValueT>> NewInitEnv(const Frame<ValueT>& frame) {
    auto ptr = Environment<ValueT>::NewInitEnv(frame);
    weak_envs_.push_back(ptr);
    return ptr;
  }

  void ClearAllFrames() {
    for (const auto& weak_env : weak_envs_) {
      if (auto env = weak_env.lock()) {
        env->ClearFrame();
      }
    }
    weak_envs_.clear();
  }

 private:
  std::list<std::weak_ptr<Environment<ValueT>>> weak_envs_;
};

}  // namespace ap::axpr
