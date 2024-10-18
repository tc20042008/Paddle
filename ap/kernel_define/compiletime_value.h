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
#include "ap/axpr/dim_expr.h"
#include "ap/axpr/value.h"
#include "ap/index_expr/index_expr.h"
#include "ap/index_expr/index_tuple_expr.h"
#include "ap/ir_match/op_match_ctx.h"
#include "ap/ir_match/tensor_match_ctx.h"
#include "ap/kernel_define/adt.h"
#include "ap/kernel_define/data_type.h"
#include "ap/kernel_define/define_ctx.h"
#include "ap/kernel_define/func_declare.h"
#include "ap/kernel_define/kernel_arg.h"
#include "ap/kernel_define/module.h"
#include "ap/kernel_define/source_code.h"
#include "paddle/cinn/adt/adt.h"

namespace ap::kernel_define {

namespace adt = ::cinn::adt;

template <typename ValueT, typename IrNodeT>
using CtValueImpl = ap::axpr::ValueBase<ValueT,
                                        axpr::DataType,
                                        axpr::PointerType,
                                        index_expr::Slice,
                                        index_expr::IndexExpr,
                                        index_expr::IndexTupleExpr,
                                        ::symbol::DimExpr,
                                        typename IrNodeT::native_op_type,
                                        typename IrNodeT::packed_op_type,
                                        typename IrNodeT::native_value_type,
                                        typename IrNodeT::packed_value_type,
                                        ir_match::OpMatchCtx<IrNodeT>,
                                        ir_match::TensorMatchCtx<IrNodeT>,
                                        DefineCtx<IrNodeT>,
                                        FuncDeclare,
                                        SourceCode,
                                        Module>;

// compile time value
template <typename IrNodeT>
struct CtValue : public CtValueImpl<CtValue<IrNodeT>, IrNodeT> {
  using CtValueImpl<CtValue<IrNodeT>, IrNodeT>::CtValueImpl;
  using ir_node_type = IrNodeT;
  DEFINE_ADT_VARIANT_METHODS(CtValueImpl<CtValue<IrNodeT>, IrNodeT>);

  static axpr::Object<CtValue<IrNodeT>> GetExportedTypes() {
    return axpr::GetObjectTypeName2Type<CtValue<IrNodeT>,
                                        axpr::DataType,
                                        axpr::PointerType,
                                        typename IrNodeT::dim_expr_type,
                                        index_expr::Slice,
                                        index_expr::IndexExpr,
                                        index_expr::IndexTupleExpr,
                                        FuncDeclare,
                                        SourceCode,
                                        Module>();
  }
};

}  // namespace ap::kernel_define
