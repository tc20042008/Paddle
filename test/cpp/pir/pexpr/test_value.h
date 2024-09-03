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

namespace pexpr::tests {

struct TestCustomValue : public std::monostate {};

using Val = Value<TestCustomValue>;

adt::Result<Val> CustomGetAttr(const TestCustomValue& val, const std::string&) {
  return TypeError{"Not implemented error."};
}

adt::Result<Val> CustomGetItem(const TestCustomValue& val, const Val& idx) {
  return TypeError{"Not implemented error."};
}

}  // namespace pexpr::tests
