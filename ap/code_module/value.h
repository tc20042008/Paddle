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
#include "ap/axpr/data_type.h"
#include "ap/axpr/pointer_type.h"
#include "ap/axpr/value.h"
#include "ap/code_module/adt.h"
#include "ap/code_module/data_type.h"
#include "ap/code_module/func_declare.h"
#include "ap/code_module/module.h"
#include "ap/code_module/source_code.h"
#include "paddle/cinn/adt/adt.h"

namespace ap::code_module {

namespace adt = ::cinn::adt;

template <typename ValueT>
using ValueImpl = ap::axpr::ValueBase<ValueT,
                                      axpr::DataType,
                                      axpr::PointerType,
                                      FuncDeclare,
                                      SourceCode,
                                      Module>;

struct Value : public ValueImpl<Value> {
  using ValueImpl<Value>::ValueImpl;
  DEFINE_ADT_VARIANT_METHODS(ValueImpl<Value>);

  static axpr::Attribute<Value> GetExportedTypes() {
    return axpr::GetObjectTypeName2Type<Value,
                                        axpr::DataType,
                                        axpr::PointerType,
                                        FuncDeclare,
                                        SourceCode,
                                        Module>();
  }
};

}  // namespace ap::code_module
