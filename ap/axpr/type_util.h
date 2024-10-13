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

#include "ap/axpr/object.h"
#include "ap/axpr/type.h"

namespace ap::axpr {

namespace detail {

template <typename ValueT, typename... ValueImplTypes>
struct GetTypeName2TypeHelper;

template <typename ValueT>
struct GetTypeName2TypeHelper<ValueT> {
  static void Call(Object<ValueT>*) {}
};

template <typename ValueT, typename ValueImplType0, typename... ValueImplTypes>
struct GetTypeName2TypeHelper<ValueT, ValueImplType0, ValueImplTypes...> {
  static void Call(Object<ValueT>* ret) {
    TypeImpl<ValueImplType0> type_impl{};
    ValueT type{type_impl};
    (*ret)->Set(type_impl.Name(), type);
    GetTypeName2TypeHelper<ValueT, ValueImplTypes...>::Call(ret);
  }
};

}  // namespace detail

template <typename ValueT, typename... ValueImplTypes>
Object<ValueT> GetObjectTypeName2Type() {
  Object<ValueT> object;
  detail::GetTypeName2TypeHelper<ValueT,
                                 Nothing,
                                 bool,
                                 int64_t,
                                 double,
                                 std::string,
                                 ValueImplTypes...>::Call(&object);
  return object;
}

}  // namespace ap::axpr
