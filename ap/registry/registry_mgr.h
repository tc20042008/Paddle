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

#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include "ap/adt/adt.h"
#include "ap/axpr/anf_expr_util.h"
#include "ap/axpr/cps_expr_interpreter.h"
#include "ap/registry/value.h"
#include "ap/registry/value_method_class.h"

namespace ap::registry {

struct RegistryMgr {
  static RegistryMgr* Singleton() {
    static RegistryMgr mgr{};
    return &mgr;
  }

  adt::Result<adt::Ok> LoadAllOnce() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!load_result_.has_value()) {
      load_result_ = VisitEachConfigFilePath(
          [&](const auto& filepath) { return Load(filepath); });
    }
    return load_result_.value();
  }

 private:
  std::optional<adt::Result<adt::Ok>> load_result_;
  std::mutex mutex_;

  adt::Result<adt::Ok> Load(const std::string& filepath) {
    ADT_LET_CONST_REF(file_content, GetFileContent(filepath));
    if (file_content.empty()) {
      return adt::Ok{};
    }
    ADT_LET_CONST_REF(anf_expr, axpr::MakeAnfExprFromJsonString(file_content));
    const auto& core_expr = axpr::ConvertAnfExprToCoreExpr(anf_expr);
    ADT_LET_CONST_REF(atomic, core_expr.TryGet<axpr::Atomic<axpr::CoreExpr>>());
    ADT_LET_CONST_REF(lambda, atomic.TryGet<axpr::Lambda<axpr::CoreExpr>>());
    axpr::CpsExprInterpreter<registry::Val> cps_expr_interpreter{};
    ADT_RETURN_IF_ERR(cps_expr_interpreter.Interpret(lambda, {}));
    return adt::Ok{};
  }

  adt::Result<std::string> GetFileContent(const std::string& filepath) {
    std::ifstream ifs(filepath);
    std::string content{std::istreambuf_iterator<char>(ifs),
                        std::istreambuf_iterator<char>()};
    return content;
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitEachConfigFilePath(const DoEachT& DoEach) {
    std::string ap_path(std::getenv("AP_PATH"));
    if (ap_path.empty()) {
      return adt::Ok{};
    }
    std::string path;
    std::istringstream ss(ap_path);
    while (std::getline(ss, path, ':')) {
      if (!path.empty()) {
        return DoEach(path);
      }
    }
    return adt::Ok{};
  }
};

}  // namespace ap::registry
