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

#include "ap/axpr/adt.h"
#include "ap/axpr/error.h"
#include "ap/axpr/type.h"
#include "ap/memory/circlable_ref.h"

namespace ap::axpr {

template <typename T>
struct MutableList
    : public memory::CirclableRef<MutableList<T>, std::vector<T>> {
  using Base = memory::CirclableRef<MutableList<T>, std::vector<T>>;
  using Base::CirclableRef;

  static MutableList Make(
      const std::shared_ptr<memory::CirclableRefListBase>& ref_list,
      std::initializer_list<T> ilist) {
    std::shared_ptr<std::vector<T>> obj(new std::vector<T>(ilist));
    return Base::Make(ref_list, obj);
  }

  static adt::Result<MutableList<T>> ThreadLocalTracked(
      std::initializer_list<T> ilist) {
    std::shared_ptr<std::vector<T>> obj(new std::vector<T>(ilist));
    return Base::ThreadLocalTracked(obj);
  }

  static adt::Result<MutableList<T>> ThreadLocalTracked() {
    return ThreadLocalTracked({});
  }
};

template <typename ValueT>
struct TypeImpl<MutableList<ValueT>> : public std::monostate {
  using value_type = MutableList<ValueT>;

  const char* Name() const { return "MutableList"; }
};

}  // namespace ap::axpr
