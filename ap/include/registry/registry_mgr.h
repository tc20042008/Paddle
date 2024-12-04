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
#include "ap/include/adt/adt.h"
#include "ap/include/axpr/anf_expr_util.h"
#include "ap/include/axpr/cps_interpreter.h"
#include "ap/include/axpr/function.h"
#include "ap/include/axpr/module_mgr.h"
#include "ap/include/axpr/serializable_value.h"
#include "ap/include/registry/builtin_frame_util.h"
#include "ap/include/registry/value.h"

namespace ap::registry {

struct RegistryMgr {
  static RegistryMgr* Singleton() {
    static RegistryMgr mgr{};
    return &mgr;
  }

  adt::Result<adt::Ok> LoadAllOnce() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!load_result_.has_value()) {
      ADT_LET_CONST_REF(filepath, GetApEntryFilePath());
      load_result_ = Load(filepath);
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
    const auto& frame = axpr::Frame<axpr::SerializableValue>::Make(
        axpr::ModuleMgr::Singleton()->circlable_ref_list(),
        std::make_shared<axpr::AttributeImpl<axpr::SerializableValue>>());
    std::vector<axpr::tVar<std::string>> args{};
    axpr::Lambda<axpr::CoreExpr> lambda{args, core_expr};
    axpr::CpsInterpreter<registry::Val> cps_expr_interpreter(
        registry::MakeBuiltinFrameAttrMap<registry::Val>());
    ADT_RETURN_IF_ERR(cps_expr_interpreter.InterpretModule(frame, lambda));
    return adt::Ok{};
  }

  adt::Result<std::string> GetFileContent(const std::string& filepath) {
    std::ifstream ifs(filepath);
    std::string content{std::istreambuf_iterator<char>(ifs),
                        std::istreambuf_iterator<char>()};
    return content;
  }

  adt::Result<std::string> GetApEntryFilePath() {
    const char* ap_entry_chars = std::getenv("AP_ENTRY");
    ADT_CHECK(ap_entry_chars != nullptr);
    std::string ap_path(ap_entry_chars);
    ADT_CHECK(FileExists(ap_path));
    return ap_path;
  }

  bool FileExists(const std::string& filepath) {
    std::fstream fp;
    fp.open(filepath, std::fstream::in);
    if (fp.is_open()) {
      fp.close();
      return true;
    } else {
      return false;
    }
  }
};

}  // namespace ap::registry
