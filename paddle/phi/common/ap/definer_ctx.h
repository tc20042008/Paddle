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
#include "paddle/cinn/adt/adt.h"
#include "paddle/phi/common/ap/adt.h"
#include "paddle/phi/common/ap/data_type.h"
#include "paddle/phi/common/ap/definer_raw_ctx.h"
#include "paddle/pir/include/dialect/pexpr/object.h"
#include "paddle/pir/include/dialect/pexpr/type.h"

namespace ap::kernel_define {

template <typename ValueT>
struct DefinerCtxImpl {
  DefinerRawCtx raw_ctx;
  pexpr::Object<ValueT> objects;

  bool operator==(const DefinerCtxImpl& other) const {
    return other.raw_ctx == this->raw_ctx && other.objects == this->objects;
  }
};

template <typename ValueT>
DEFINE_ADT_RC(DefinerCtx, DefinerCtxImpl<ValueT>);

}  // namespace ap::kernel_define

namespace pexpr {

template <typename ValueT>
struct TypeImpl<ap::kernel_define::DefinerCtx<ValueT>> : public std::monostate {
  using value_type = ap::kernel_define::DefinerCtx<ValueT>;

  const char* Name() const { return "DefinerCtx"; }
};

}  // namespace pexpr
