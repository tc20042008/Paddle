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
#include "ap/axpr/builtin_serializable_object.h"
#include "ap/axpr/dim_expr.h"
#include "ap/axpr/value.h"
#include "ap/code_gen/code_gen_ctx.h"
#include "ap/code_gen/code_gen_result.h"
#include "ap/code_gen/dim_expr_kernel_arg_id.h"
#include "ap/code_gen/in_tensor_data_ptr_kernel_arg_id.h"
#include "ap/code_gen/out_tensor_data_ptr_kernel_arg_id.h"
#include "ap/code_module/adt.h"
#include "ap/code_module/data_type.h"
#include "ap/code_module/func_declare.h"
#include "ap/code_module/module.h"
#include "ap/code_module/source_code.h"
#include "ap/index_expr/index_expr.h"
#include "ap/index_expr/index_tuple_expr.h"
#include "ap/ir_match/op_match_ctx.h"
#include "ap/ir_match/tensor_match_ctx.h"
#include "paddle/cinn/adt/adt.h"

namespace ap::code_gen {

namespace adt = ::cinn::adt;

template <typename ValueT, typename BirNode>
using ValueImpl = ap::axpr::ValueBase<ValueT,
                                      axpr::DataType,
                                      axpr::PointerType,
                                      axpr::BuiltinSerializableObject<ValueT>,
                                      index_expr::Slice,
                                      index_expr::IndexExpr,
                                      index_expr::IndexTupleExpr,
                                      ::symbol::DimExpr,
                                      typename BirNode::native_op_type,
                                      typename BirNode::packed_op_type,
                                      typename BirNode::ref_op_type,
                                      typename BirNode::native_value_type,
                                      typename BirNode::ref_value_type,
                                      DimExprKernelArgId<BirNode>,
                                      InTensorDataPtrKernelArgId<BirNode>,
                                      OutTensorDataPtrKernelArgId<BirNode>,
                                      ir_match::OpMatchCtx<BirNode>,
                                      ir_match::TensorMatchCtx<BirNode>,
                                      CodeGenCtx<BirNode>,
                                      code_module::FuncDeclare,
                                      code_module::SourceCode,
                                      code_module::Module,
                                      CodeGenResult<ValueT>>;

// compile time value
template <typename BirNode>
struct Value : public ValueImpl<Value<BirNode>, BirNode> {
  using ValueImpl<Value<BirNode>, BirNode>::ValueImpl;
  using ir_node_type = BirNode;
  DEFINE_ADT_VARIANT_METHODS(ValueImpl<Value<BirNode>, BirNode>);

  static axpr::Object<Value<BirNode>> GetExportedTypes() {
    return axpr::GetObjectTypeName2Type<
        Value<BirNode>,
        axpr::DataType,
        axpr::PointerType,
        axpr::BuiltinSerializableObject<Value<BirNode>>,
        typename BirNode::dim_expr_type,
        index_expr::Slice,
        index_expr::IndexExpr,
        index_expr::IndexTupleExpr,
        code_module::FuncDeclare,
        code_module::SourceCode,
        code_module::Module,
        CodeGenResult<Value<BirNode>>>();
  }
};

}  // namespace ap::code_gen
