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

#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/naive_class_ops.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/code_module/directory.h"
#include "paddle/ap/include/code_module/file.h"

namespace ap::code_module {

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetDirectoryClass();

struct TypeDirectoryClassMethodClass {
  static adt::Result<axpr::Value> New(
      const axpr::Value&, const std::vector<axpr::Value>& args_vec) {
    const auto& packed_args = axpr::CastToPackedArgs(args_vec);
    const auto& [args, kwargs] = *packed_args;
    ADT_CHECK(args->empty()) << adt::errors::TypeError{
        std::string() + "Directory() takes no positional argument, bug " +
        std::to_string(args->size()) + " were given"};
    axpr::AttrMap<File> dentry2file{};
    for (const auto& [dentry, elt] : kwargs->storage) {
      ADT_LET_CONST_REF(file, File::CastFromAxprValue(elt))
          << adt::errors::TypeError{
                 std::string() +
                 "Directory() only acepts Project.FileContent, "
                 "Project.SoftLink and Project.Directory, but " +
                 axpr::GetTypeName(elt) + " were given"};
      dentry2file->Set(dentry, file);
    }
    return GetDirectoryClass().New(Directory<File>{dentry2file});
  }
};

inline axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetDirectoryClass() {
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>("Directory", [&](const auto& DoEach) {
        DoEach("__init__", &TypeDirectoryClassMethodClass::New);
      }));
  return axpr::MakeGlobalNaiveClassOps<Directory<File>>(cls);
}

}  // namespace ap::code_module
