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

namespace pexpr {

#define PEXPR_FOR_EACH_BINARY_OP(_) \
  _(Add, +)                         \
  _(Sub, -)                         \
  _(Mul, *)                         \
  _(Div, /)                         \
  _(Mod, %)                         \
  _(EQ, ==)                         \
  _(NE, !=)                         \
  _(GT, >)                          \
  _(GE, >=)                         \
  _(LT, <)                          \
  _(LE, <=)

}  // namespace pexpr
