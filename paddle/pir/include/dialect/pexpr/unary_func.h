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

#define PEXPR_FOR_EACH_UNARY_OP(_) \
  _(Not, !)                        \
  _(Neg, -)

#define DEFINE_ARITHMETIC_UNARY_OP(name, op)            \
  struct Arithmetic##name {                             \
    static constexpr const char* Name() { return #op; } \
                                                        \
    template <typename LhsT>                            \
    static auto Call(const LhsT& val) {                 \
      return op val;                                    \
    }                                                   \
  };
PEXPR_FOR_EACH_UNARY_OP(DEFINE_ARITHMETIC_UNARY_OP);
#undef DEFINE_ARITHMETIC_UNARY_OP

}  // namespace pexpr
