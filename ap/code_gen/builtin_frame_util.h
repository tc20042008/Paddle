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

#include "ap/adt/adt.h"
#include "ap/axpr/builtin_frame_util.h"
#include "ap/axpr/dim_expr_method_class.h"
#include "ap/code_module/func_declare_method_class.h"
#include "ap/code_module/module_method_class.h"
#include "ap/code_module/source_code_method_class.h"
#include "ap/index_expr/index_expr_method_class.h"
#include "ap/index_expr/index_tuple_expr_method_class.h"
#include "ap/index_expr/slice_method_class.h"

namespace ap::code_gen {

template <typename ValueT, typename DoEachT>
void VisitEachBuiltinFrameClass(const DoEachT& DoEach) {
  DoEach(code_module::MakeSourceCodeClass<ValueT>());
  DoEach(code_module::MakeFuncDeclareClass<ValueT>());
  DoEach(code_module::MakeModuleClass<ValueT>());
  DoEach(axpr::GetDimExprClass<ValueT>());
  DoEach(index_expr::GetSliceClass<ValueT>());
  DoEach(index_expr::GetIndexExprClass<ValueT>());
  DoEach(index_expr::GetIndexTupleExprClass<ValueT>());
}

template <typename ValueT>
axpr::AttrMap<ValueT> MakeBuiltinFrameAttrMap() {
  axpr::AttrMap<ValueT> attr_map;
  axpr::VisitEachBuiltinFrameAttr<ValueT>(
      [&](const std::string& k, const ValueT& v) { attr_map->Set(k, v); });
  VisitEachBuiltinFrameClass<ValueT>(
      [&](const auto& cls) { attr_map->Set(cls.Name(), cls); });
  return attr_map;
}

}  // namespace ap::code_gen
