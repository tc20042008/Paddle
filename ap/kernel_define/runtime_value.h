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
#include "ap/kernel_define/adt.h"
#include "ap/kernel_define/data_type.h"
#include "ap/kernel_define/func_declare.h"
#include "ap/kernel_define/kernel_arg.h"
#include "ap/kernel_define/module.h"
#include "ap/kernel_define/source_code.h"
#include "ap/kernel_define/undefined_ir_node.h"
#include "paddle/cinn/adt/adt.h"

namespace ap::kernel_define {

namespace adt = ::cinn::adt;

template <typename ValueT, typename IrNodeT>
using RtValueImpl = ap::axpr::ValueBase<ValueT,
                                        axpr::DataType,
                                        axpr::PointerType,
                                        KernelArg,
                                        FuncDeclare,
                                        SourceCode,
                                        Module>;

template <typename IrNodeT>
struct RtValue : public RtValueImpl<RtValue<IrNodeT>, IrNodeT> {
  using RtValueImpl<RtValue<IrNodeT>, IrNodeT>::RtValueImpl;
  using ir_node_type = IrNodeT;
  DEFINE_ADT_VARIANT_METHODS(RtValueImpl<RtValue<IrNodeT>, IrNodeT>);

  static axpr::Object<RtValue<IrNodeT>> GetExportedTypes() {
    return axpr::GetObjectTypeName2Type<RtValue<IrNodeT>,
                                        axpr::DataType,
                                        axpr::PointerType,
                                        KernelArg,
                                        FuncDeclare,
                                        SourceCode,
                                        Module>();
  }
};

}  // namespace ap::kernel_define
