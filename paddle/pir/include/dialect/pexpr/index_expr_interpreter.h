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
#include "paddle/pir/include/dialect/pexpr/builtin_functions.h"
#include "paddle/pir/include/dialect/pexpr/core_expr.h"
#include "paddle/pir/include/dialect/pexpr/index_expr.h"
#include "paddle/pir/include/dialect/pexpr/index_expr_builtin_functions.h"
#include "paddle/pir/include/dialect/pexpr/index_expr_value.h"

namespace pexpr::index_expr {

class IndexExprInterpreterImpl;

class IndexExprInterpreter {
 public:
  IndexExprInterpreter();
  explicit IndexExprInterpreter(const std::shared_ptr<EnvMgr>& env_mgr);
  IndexExprInterpreter(const IndexExprInterpreter&) = delete;
  IndexExprInterpreter(IndexExprInterpreter&&) = delete;

  Result<Val> operator()(const Lambda<CoreExpr>& lambda,
                         const std::vector<Val>& args);

  Result<Val> operator()(
      const std::unordered_map<std::string, BuiltinFuncType<Val>>&
          global_functions,
      const Lambda<CoreExpr>& lambda,
      const std::vector<Val>& args);

 private:
  std::shared_ptr<EnvMgr> env_mgr_;
  std::shared_ptr<IndexExprInterpreterImpl> impl_;
};

}  // namespace pexpr::index_expr
