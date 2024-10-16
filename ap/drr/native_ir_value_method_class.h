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

#include "ap/axpr/method_class.h"
#include "ap/axpr/type.h"
#include "ap/drr/native_ir_value.h"
#include "ap/drr/tags.h"

namespace ap::axpr {

template <typename ValueT, typename NodeT>
struct MethodClassImpl<ValueT, drr::tSrcPtn<drr::NativeIrValue<NodeT>>> {
  using Self = drr::tSrcPtn<drr::NativeIrValue<NodeT>>;
  using This = MethodClassImpl<ValueT, Self>;

  adt::Result<ValueT> ToString(const Self& self) {
    std::ostringstream ss;
    const void* ptr = self.value().__adt_rc_shared_ptr_raw_ptr();
    ss << "<" << axpr::TypeImpl<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  adt::Result<ValueT> Starred(const Self& self) {
    return adt::errors::TypeError{
        std::string() +
        "Only SrcPtnPackedIrValue and ResPtnPackedIrValue tensors can be "
        "unpacked. tensor '" +
        self.value()->name + "' is of type 'SrcPtnNativeIrValue'"};
  }
};

template <typename ValueT, typename NodeT>
struct MethodClassImpl<ValueT,
                       TypeImpl<drr::tSrcPtn<drr::NativeIrValue<NodeT>>>> {};

template <typename ValueT, typename NodeT>
struct MethodClassImpl<ValueT, drr::tResPtn<drr::NativeIrValue<NodeT>>> {
  using Self = drr::tResPtn<drr::NativeIrValue<NodeT>>;
  using This = MethodClassImpl<ValueT, Self>;

  adt::Result<ValueT> ToString(const Self& self) {
    std::ostringstream ss;
    const void* ptr = self.value().__adt_rc_shared_ptr_raw_ptr();
    ss << "<" << axpr::TypeImpl<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  adt::Result<ValueT> Starred(const Self& self) {
    return adt::errors::TypeError{
        std::string() +
        "Only SrcPtnPackedIrValue and ResPtnPackedIrValue tensors can be "
        "unpacked. tensor '" +
        self.value()->name + "' is of type 'ResPtnNativeIrValue'"};
  }
};

template <typename ValueT, typename NodeT>
struct MethodClassImpl<ValueT,
                       TypeImpl<drr::tResPtn<drr::NativeIrValue<NodeT>>>> {};

}  // namespace ap::axpr
