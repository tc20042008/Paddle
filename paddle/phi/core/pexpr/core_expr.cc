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

#include "paddle/phi/core/pexpr/core_expr.h"
#include <iomanip>
#include <sstream>

namespace pexpr {

namespace {

std::string AtomicExprToSExpression(const Atomic<CoreExpr>& core_expr) {
  return core_expr.Match(
      [](const tVar<std::string>& var) { return var.value(); },
      [](const bool c) {
        return c ? std::string("#t") : std::string("#false");
      },
      [](const int64_t c) { return std::to_string(c); },
      [](const std::string& str) {
        std::ostringstream ss;
        ss << std::quoted(str);
        return ss.str();
      },
      [](const PrimitiveOp& op) {
        return std::string("(op ") + op.op_name + ")";
      },
      [](const Lambda<CoreExpr>& lambda) {
        std::ostringstream ss;
        ss << "(lambda [";
        int i = 0;
        for (const auto& arg : lambda.args) {
          if (i++ > 0) {
            ss << " ";
          }
          ss << arg.value();
        }
        ss << "] ";
        ss << lambda.body->ToSExpression();
        ss << ")";
        return ss.str();
      });
}

std::string ComposedCallExprToSExpression(
    const ComposedCall<CoreExpr>& core_expr) {
  std::ostringstream ss;
  ss << "(";
  ss << AtomicExprToSExpression(core_expr.outter_func);
  ss << " ";
  ss << AtomicExprToSExpression(core_expr.inner_func);
  for (const auto& arg : core_expr.args) {
    ss << " ";
    ss << AtomicExprToSExpression(arg);
  }
  ss << ")";
  return ss.str();
}

}  // namespace

std::string CoreExpr::ToSExpression() const {
  return Match(
      [&](const Atomic<CoreExpr>& core_expr) {
        return AtomicExprToSExpression(core_expr);
      },
      [&](const ComposedCall<CoreExpr>& core_expr) {
        return ComposedCallExprToSExpression(core_expr);
      });
}

}  // namespace pexpr
