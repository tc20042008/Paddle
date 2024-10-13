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

#include "ap/axpr/method_class.h"
#include "ap/axpr/packed_args.h"
#include "ap/index_expr/index_tuple_expr.h"
#include "ap/kernel_define/cuda_code_gen_util.h"
#include "ap/kernel_define/ir_op.h"
#include "ap/kernel_define/module.h"
#include "ap/kernel_define/op_code_gen_ctx.h"

namespace ap::kernel_define {

using ap::axpr::BuiltinBinaryFuncT;
using ap::axpr::BuiltinFuncType;
using ap::axpr::BuiltinUnaryFuncT;
using ap::axpr::CppDataType;
using ap::axpr::CppPointerType;
using ap::axpr::DataType;
using ap::axpr::MethodClass;
using ap::axpr::PointerType;

template <typename ValueT, typename IrNodeT>
struct DefineCtxMethodClass {
  using This = DefineCtxMethodClass;
  using Self = DefineCtx<IrNodeT>;

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, axpr::TryGetImpl<std::string>(attr_name_val));
    if (attr_name == "match_ctx") {
      return GetMatchCtx(self);
    }
    if (attr_name == "cuda_code_gen") {
      return &This::StaticCudaCodeGen;
    }
    return adt::errors::AttributeError{
        std::string("'DefineCtx' object has no attribute '") + attr_name +
        "' "};
  }

  static adt::Result<ValueT> StaticCudaCodeGen(
      const ValueT& self_val, const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, axpr::TryGetImpl<Self>(self_val));
    return This{}.CudaCodeGen(self, args);
  }

  adt::Result<ValueT> CudaCodeGen(const Self& self,
                                  const std::vector<ValueT>& packed_args_vec) {
    ADT_LET_CONST_REF(packed_args, axpr::CastToPackedArgs(packed_args_vec))
        << adt::errors::TypeError{
               "'DefineCtx.cuda_code_gen' has no keyword arguments."};
    const auto& [args, kwargs] = *packed_args;
    ADT_CHECK(args->size() == 1) << adt::errors::TypeError{
        "'DefineCtx.cuda_code_gen' takes 1 positional arguments but " +
        std::to_string(args->size()) + " were given."};
    ADT_LET_CONST_REF(ir_op, IrOp<IrNodeT>::CastFrom(args->at(0)))
        << adt::errors::TypeError{
               std::string() +
               "the positional argument 1 of 'DefineCtx.cuda_code_gen' should "
               "be able to cast to a NativeIrOp or PackedIrOp."};
    ADT_LET_CONST_REF(loop_index_tuple_expr,
                      kwargs->template Get<index_expr::IndexTupleExpr>(
                          "loop_index_tuple_expr"))
        << adt::errors::TypeError{
               std::string() +
               "'DefineCtx.cuda_code_gen' requires 'IndexTupleExpr' typed "
               "keyword argument 'loop_index_tuple_expr'."};
    std::vector<std::string> loop_var_names{};
    {
      ADT_LET_CONST_REF(
          loop_var_names_val,
          kwargs->template Get<adt::List<ValueT>>("loop_var_names"))
          << adt::errors::TypeError{std::string() +
                                    "'DefineCtx.cuda_code_gen' requires 'list' "
                                    "typed keyword argument 'loop_var_names'."};
      loop_var_names.reserve(loop_var_names_val->size());
      for (const auto& elt : *loop_var_names_val) {
        ADT_LET_CONST_REF(loop_var_name, axpr::TryGetImpl<std::string>(elt))
            << adt::errors::TypeError{
                   std::string() +
                   "keyword argument 'loop_var_names' of "
                   "'DefineCtx.cuda_code_gen' should be a list of string."};
        loop_var_names.emplace_back(loop_var_name);
      }
    }
    ADT_LET_CONST_REF(
        anchor_local_var_name,
        kwargs->template GetOpt<std::string>("anchor_local_var_name"))
        << adt::errors::TypeError{std::string() +
                                  "keyword argument 'anchor_local_var_name' of "
                                  "'DefineCtx.cuda_code_gen' should be a str"};
    std::vector<LocalVarBinding<IrNodeT>> local_var_name_bindings{};
    {
      ADT_LET_CONST_REF(
          local_var_name_bindings_val,
          kwargs->template Get<adt::List<ValueT>>("local_var_name_bindings"))
          << adt::errors::TypeError{
                 std::string() +
                 "'DefineCtx.cuda_code_gen' requires 'list' typed keyword "
                 "argument 'local_var_name_bindings'."};
      local_var_name_bindings.reserve(local_var_name_bindings_val->size());
      for (const auto& elt : *local_var_name_bindings_val) {
        ADT_LET_CONST_REF(pair, axpr::TryGetImpl<adt::List<ValueT>>(elt))
            << adt::errors::TypeError{
                   std::string() +
                   "keyword argument 'local_var_name_bindings' of "
                   "'DefineCtx.cuda_code_gen' should be a list of pair."};
        ADT_CHECK(pair->size() == 2) << adt::errors::TypeError{
            std::string() +
            "keyword argument 'local_var_name_bindings' of "
            "'DefineCtx.cuda_code_gen' should be a list of pair."};
        ADT_LET_CONST_REF(local_var_name,
                          axpr::TryGetImpl<std::string>(pair->at(0)))
            << adt::errors::TypeError{
                   std::string() +
                   "keyword argument 'local_var_name_bindings' of "
                   "'DefineCtx.cuda_code_gen' should be a list of pair(str, "
                   "NativeIrValue)."};
        using NativeIrValue = typename IrNodeT::native_value_type;
        ADT_LET_CONST_REF(ir_tensor,
                          axpr::TryGetImpl<NativeIrValue>(pair->at(1)))
            << adt::errors::TypeError{
                   std::string() +
                   "keyword argument 'local_var_name_bindings' of "
                   "'DefineCtx.cuda_code_gen' should be a list of pair(str, "
                   "NativeIrValue)."};
        LocalVarBinding<IrNodeT> binding{local_var_name, ir_tensor};
        local_var_name_bindings.emplace_back(binding);
      }
    }
    OpCodeGenCtx<IrNodeT> op_code_gen_ctx{
        self.shared_ptr(),
        loop_index_tuple_expr,
        loop_var_names,
        local_var_name_bindings,
        anchor_local_var_name,
    };
    ADT_LET_CONST_REF(code_str, OpCudaCodeGen<IrNodeT>(op_code_gen_ctx, ir_op));
    return code_str;
  }

  adt::Result<ValueT> GetMatchCtx(const Self& self) {
    ADT_CHECK(self->ir_match_ctx.has_value()) << adt::errors::ValueError{
        "'DefineCtx.ir_match_ctx' has not been initialized."};
    return self->ir_match_ctx.value();
  }
};

}  // namespace ap::kernel_define

namespace ap::axpr {

template <typename ValueT, typename IrNodeT>
struct MethodClassImpl<ValueT, ap::kernel_define::DefineCtx<IrNodeT>>
    : public ap::kernel_define::DefineCtxMethodClass<ValueT, IrNodeT> {};

template <typename ValueT, typename IrNodeT>
struct MethodClassImpl<ValueT, TypeImpl<ap::kernel_define::DefineCtx<IrNodeT>>>
    : public EmptyMethodClass<ValueT> {};

}  // namespace ap::axpr
