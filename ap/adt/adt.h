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

#include <unordered_set>
#include "llvm/ADT/SmallVector.h"
#include "paddle/cinn/adt/adt.h"

namespace adt = ::cinn::adt;

namespace ap {

static constexpr int kSmallFeatureSize() { return 11; }

template <typename T>
using SmallVector = llvm::SmallVector<T, kSmallFeatureSize()>;

}  // namespace ap
