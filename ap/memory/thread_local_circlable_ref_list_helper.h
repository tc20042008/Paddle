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
#include "ap/memory/circlable_ref_guarded.h"
#include "ap/memory/circlable_ref_list.h"

namespace ap::memory {

struct ThreadLocalCirclableRefListHelper {
 private:
  using This = ThreadLocalCirclableRefListHelper;

  using OptList = std::optional<std::shared_ptr<CirclableRefList>>;

  static OptList* MutThreadLocalList() {
    thread_local static OptList list;
    return &list;
  }

  struct Scope {
    explicit Scope(const std::shared_ptr<CirclableRefList>& new_list) {
      OptList old_list = *MutThreadLocalList();
      *MutThreadLocalList() = new_list;
    }
    ~Scope() { *MutThreadLocalList() = old_list; }
    OptList old_list;
  };

 public:
  template <typename CallbackT>
  static auto Guard(const CallbackT& Callback) {
    Scope scope{std::make_shared<CirclableRefList>()};
    return Callback();
  }

  static const OptList& Current() { return *MutThreadLocalList(); }

  template <typename T>
  static adt::Result<CirclableRefGuarded<T>> MakeGuarded(const T& obj) {
    ADT_CHECK(Current().has_value()) << adt::errors::RuntimeError{
        "no ThreadLocalCirclableRefListGuard found in current thread."};
    return CirclableRefGuarded<T>(Current().value(), obj);
  }
};

}  // namespace ap::memory
