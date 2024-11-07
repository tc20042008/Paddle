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
#include <unordered_map>
#include <utility>
#include "ap/adt/adt.h"
#include "ap/axpr/hash.h"
#include "ap/axpr/type.h"

namespace ap::axpr {

template <typename ValueT, typename Hasher>
struct OrderedSetImpl {
 public:
  OrderedSetImpl() {}

  bool operator==(const OrderedSetImpl& other) const {
    return this->items() == other.items();
  }

  const std::list<ValueT>& items() const { return items_; }

  adt::Result<bool> Has(const ValueT& key) const {
    Hasher hasher{};
    ADT_LET_CONST_REF(hash_value, hasher(key));
    const auto& iter_to_iters = this->hash_value2pair_iters_.find(key);
    if (iter_to_iters == this->hash_value2pair_iters_.end()) {
      return false;
    }
    for (auto iter : iter_to_iters->second) {
      if (*iter == key) {
        return true;
      }
    }
    return false;
  }

  adt::Result<adt::Ok> Insert(const ValueT& val) {
    Hasher hasher{};
    ADT_LET_CONST_REF(hash_value, hasher(val));
    auto* lst = &this->hash_value2pair_iters_[hash_value];
    for (auto iter : *lst) {
      if (*iter == val) {
        *iter = val;
        return adt::Ok{};
      }
    }
    lst->emplace_back(this->items_.insert(this->items_.end(), val));
    return adt::Ok{};
  }

 private:
  using ItemsT = std::list<ValueT>;
  ItemsT items_;
  std::unordered_map<int64_t, std::list<typename ItemsT::iterator>>
      hash_value2pair_iters_;
};

template <typename ValueT>
DEFINE_ADT_RC(OrderedSet, OrderedSetImpl<ValueT, axpr::Hash<ValueT>>);

template <typename ValueT>
struct TypeImpl<OrderedSet<ValueT>> : public std::monostate {
  using std::monostate::monostate;

  const char* Name() const { return "OrderedSet"; }
};

}  // namespace ap::axpr
