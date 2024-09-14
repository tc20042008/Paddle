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
#include "ap/axpr/adt.h"
#include "ap/axpr/frame.h"
#include "ap/axpr/object.h"

namespace ap::axpr {

template <typename ValueT>
class EnvironmentManager;

template <typename ValueT>
class Environment {
 public:
  Result<ValueT> Get(const std::string& var) const {
    const Result<ValueT>& res = frame_.Get(var);
    if (res.template Has<ValueT>()) {
      return res;
    }
    if (parent_ == nullptr) {
      return NameError{std::string("name '") + var + "' is not defined."};
    }
    return parent_->Get(var);
  }

  bool Set(const std::string& var, const ValueT& val) {
    return frame_.Set(var, val);
  }

  void ClearFrame() {
    parent_ = nullptr;
    frame_.ClearFrame();
  }

 private:
  static std::shared_ptr<Environment> New(
      const std::shared_ptr<Environment>& parent) {
    return std::shared_ptr<Environment>(
        new Environment(parent, Frame<ValueT>{}));
  }

  static std::shared_ptr<Environment> NewInitEnv(const Frame<ValueT>& frame) {
    return std::shared_ptr<Environment>(new Environment(nullptr, frame));
  }

  explicit Environment(const std::shared_ptr<Environment>& parent,
                       const Frame<ValueT>& frame)
      : parent_(parent), frame_(frame) {}

  Environment(const Environment&) = delete;
  Environment(Environment&&) = delete;

  friend class EnvironmentManager<ValueT>;

  std::shared_ptr<Environment> parent_;
  Frame<ValueT> frame_;
};

}  // namespace ap::axpr
