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

#include <mutex>
#include "ap/adt/adt.h"
#include "ap/registry/registry.h"

namespace ap::registry {

template <typename ValueT>
struct RegistrySingleton {
  static adt::Result<Registry<ValueT>> Singleton() {
    std::unique_lock<std::mutex> lock(*SingletonMutex());
    ADT_CHECK(MutOptSingleton()->has_value())
        << adt::errors::NotImplementedError{
               std::string() + "Registry singleton not initialized. "};
    return MutOptSingleton()->value();
  }

  static void Add(const OpIndexesExprRegistryItem& item) {
    auto registry = MutSingleton();
    const auto& op_name = item->op_name;
    int64_t nice = item->nice;
    std::unique_lock<std::mutex> lock(*SingletonMutex());
    registry->op_indexes_expr_registry_items[op_name][nice].emplace_back(item);
  }

  static void Add(const DrrRegistryItem& item) {
    auto registry = MutSingleton();
    const auto& drr_pass_name = item->drr_pass_name;
    int64_t nice = item->nice;
    std::unique_lock<std::mutex> lock(*SingletonMutex());
    registry->drr_registry_items[drr_pass_name][nice].emplace_back(item);
  }

  static void Add(const OpComputeRegistryItem& item) {
    auto registry = MutSingleton();
    const auto& op_name = item->op_name;
    int64_t nice = item->nice;
    std::unique_lock<std::mutex> lock(*SingletonMutex());
    registry->op_compute_registry_items[op_name][nice].emplace_back(item);
  }

  static std::mutex* SingletonMutex() {
    static std::mutex mutex;
    return &mutex;
  }

 private:
  static Registry<ValueT> MutSingleton() {
    std::unique_lock<std::mutex> lock(*SingletonMutex());
    if (!MutOptSingleton()->has_value()) {
      *MutOptSingleton() = Registry<ValueT>{};
    }
    return MutOptSingleton()->value();
  }

  static std::optional<Registry<ValueT>>* MutOptSingleton() {
    static std::optional<Registry<ValueT>> ctx{};
    return &ctx;
  }
};

}  // namespace ap::registry
