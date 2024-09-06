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

#include "paddle/pir/include/dialect/pexpr/value.h"
#include "paddle/pir/include/dialect/pexpr/value_method_class.h"

namespace pexpr::tests {

struct TestValue : public ValueBase<TestValue> {
  using ValueBase<TestValue>::ValueBase;
  DEFINE_ADT_VARIANT_METHODS(ValueBase<TestValue>);
}

using Val = TestValue;

}  // namespace pexpr::tests
