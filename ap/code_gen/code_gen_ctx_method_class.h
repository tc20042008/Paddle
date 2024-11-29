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
#include "ap/code_gen/arg_source_helper.h"
#include "ap/code_gen/cuda_code_gen_util.h"
#include "ap/code_gen/ir_op.h"
#include "ap/code_gen/kernel_arg_id_helper.h"
#include "ap/code_gen/op_code_gen_ctx.h"
#include "ap/code_module/module.h"
#include "ap/index_expr/index_tuple_expr.h"
#include "ap/ir_match/native_or_ref_ir_value.h"
#include "ap/registry/registry_singleton.h"

namespace ap::code_gen {

using ap::axpr::BuiltinBinaryFunc;
using ap::axpr::BuiltinFuncType;
using ap::axpr::BuiltinUnaryFunc;
using ap::axpr::CppDataType;
using ap::axpr::CppPointerType;
using ap::axpr::DataType;
using ap::axpr::MethodClass;
using ap::axpr::PointerType;

template <typename ValueT, typename BirNode>
struct CodeGenCtxMethodClass {
  using This = CodeGenCtxMethodClass;
  using Self = CodeGenCtx<BirNode>;

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, axpr::TryGetImpl<std::string>(attr_name_val));
    if (attr_name == "make_fusion_op_code_gen_class") {
      return axpr::Method<ValueT>{self, &This::StaticMakeFusionOpCodeGenClass};
    }
    if (attr_name == "dim_expr_kernel_arg_id") {
      return axpr::Method<ValueT>{self,
                                  &This::StaticMakeAndCheckDimExprKernelArgId};
    }
    if (attr_name == "in_tensor_data_ptr_kernel_arg_id") {
      return axpr::Method<ValueT>{
          self, &This::StaticMakeAndCheckInTensorDataPtrKernelArgId};
    }
    if (attr_name == "out_tensor_data_ptr_kernel_arg_id") {
      return axpr::Method<ValueT>{
          self, &This::StaticMakeAndCheckOutTensorDataPtrKernelArgId};
    }
    return adt::errors::AttributeError{
        std::string("'CodeGenCtx' object has no attribute '") + attr_name +
        "' "};
  }

  static adt::Result<ValueT> StaticMakeAndCheckOutTensorDataPtrKernelArgId(
      const ValueT& self_val, const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template TryGet<Self>());
    return This{}.MakeAndCheckOutTensorDataPtrKernelArgId(self, args);
  }

  adt::Result<ValueT> MakeAndCheckOutTensorDataPtrKernelArgId(
      const Self& self, const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1)
        << adt::errors::TypeError{std::string() +
                                  "out_tensor_data_ptr_kernel_"
                                  "arg_id() takes 1 argument but " +
                                  std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(ir_value, CastToBirValue(args.at(0)))
        << adt::errors::TypeError{
               std::string() +
               "the argument 1 of "
               "out_tensor_data_ptr_kernel_arg_id() should be "
               "'NativeIrValue' or 'RefIrValue' (not '" +
               axpr::GetTypeName(args.at(0)) + "')."};
    ADT_RETURN_IF_ERR(CheckOutTensorDataPtrRuntimeAvailable(self, ir_value));
    OutTensorDataPtrKernelArgId<BirNode> uninitialized{ir_value, std::nullopt};
    ArgSourceHelper<BirNode> helper{self->arg_source_ctx};
    ADT_LET_CONST_REF(runtime_getter,
                      helper.MakeRuntimeKerneArgGetter(uninitialized));
    return OutTensorDataPtrKernelArgId<BirNode>{ir_value, runtime_getter};
  }

  adt::Result<adt::Ok> CheckOutTensorDataPtrRuntimeAvailable(
      const Self& self, const BirNode& ir_value) {
    ADT_CHECK(self->arg_source_ctx->GetOutputTensorSource(ir_value).has_value())
        << adt::errors::TypeError{
               std::string() +
               "out_tensor_data_ptr_kernel_arg_id() failed. "
               "please check whether the ir_value is an output value of the "
               "current ap_pattern_fusion_op defined in drr result pattern "
               "lambda."};
    return adt::Ok{};
  }

  static adt::Result<ValueT> StaticMakeAndCheckInTensorDataPtrKernelArgId(
      const ValueT& self_val, const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template TryGet<Self>());
    return This{}.MakeAndCheckInTensorDataPtrKernelArgId(self, args);
  }

  adt::Result<ValueT> MakeAndCheckInTensorDataPtrKernelArgId(
      const Self& self, const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1)
        << adt::errors::TypeError{std::string() +
                                  "in_tensor_data_ptr_kernel_"
                                  "arg_id() takes 1 argument but " +
                                  std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(ir_value, CastToBirValue(args.at(0)))
        << adt::errors::TypeError{
               std::string() +
               "the argument 1 of "
               "in_tensor_data_ptr_kernel_arg_id() should be "
               "'NativeIrValue' or 'RefIrValue' (not '" +
               axpr::GetTypeName(args.at(0)) + "')."};
    ADT_RETURN_IF_ERR(CheckInTensorDataPtrRuntimeAvailable(self, ir_value));
    InTensorDataPtrKernelArgId<BirNode> uninitialized{ir_value, std::nullopt};
    ArgSourceHelper<BirNode> helper{self->arg_source_ctx};
    ADT_LET_CONST_REF(runtime_getter,
                      helper.MakeRuntimeKerneArgGetter(uninitialized));
    return InTensorDataPtrKernelArgId<BirNode>{ir_value, runtime_getter};
  }

  adt::Result<adt::Ok> CheckInTensorDataPtrRuntimeAvailable(
      const Self& self, const BirNode& ir_value) {
    ADT_CHECK(self->arg_source_ctx->GetInputTensorSource(ir_value).has_value())
        << adt::errors::TypeError{
               std::string() +
               "in_tensor_data_ptr_kernel_arg_id() failed. "
               "please check whether the ir_value is an input value of the "
               "current ap_pattern_fusion_op defined in drr result pattern "
               "lambda."};
    return adt::Ok{};
  }

  adt::Result<BirNode> CastToBirValue(const ValueT& val) {
    using RetT = adt::Result<BirNode>;
    return val.Match(
        [&](const typename BirNode::native_value_type& impl) -> RetT {
          return impl;
        },
        [&](const typename BirNode::ref_value_type& impl) -> RetT {
          return impl;
        },
        [&](const auto& impl) -> RetT {
          using T = std::decay_t<decltype(impl)>;
          return adt::errors::NotImplementedError{
              std::string() +
              "CastToBirValue() failed. only 'NativeIrValue' and 'RefIrValue' "
              "argument is expected, but '" +
              typeid(T).name() + "' found."};
        });
  }

  static adt::Result<ValueT> StaticMakeAndCheckDimExprKernelArgId(
      const ValueT& self_val, const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template TryGet<Self>());
    return This{}.MakeAndCheckDimExprKernelArgId(self, args);
  }

  adt::Result<ValueT> MakeAndCheckDimExprKernelArgId(
      const Self& self, const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "dim_expr_kernel_arg_id() takes 1 arguments but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(dim_expr, args.at(0).template TryGet<symbol::DimExpr>())
        << adt::errors::TypeError{std::string() +
                                  "the argument 1 of dim_expr_kernel_arg_id() "
                                  "should be 'DimExpr' (not '" +
                                  axpr::GetTypeName(args.at(0)) + "')."};
    ADT_RETURN_IF_ERR(CheckDimExprRuntimeAvailable(self, dim_expr));
    DimExprKernelArgId<BirNode> uninitialized{dim_expr, std::nullopt};
    ArgSourceHelper<BirNode> helper{self->arg_source_ctx};
    ADT_LET_CONST_REF(runtime_getter,
                      helper.MakeRuntimeKerneArgGetter(uninitialized));
    return DimExprKernelArgId<BirNode>{dim_expr, runtime_getter};
  }

  adt::Result<adt::Ok> CheckDimExprRuntimeAvailable(
      const Self& self, const symbol::DimExpr& dim_expr) {
    ADT_CHECK(self->arg_source_ctx->HasDirectOrIndirectDimExprSource(dim_expr))
        << adt::errors::ValueError{
               std::string() +
               "DimExpr could not evaluated in runtime. value: " +
               symbol::ToString(dim_expr)};
    return adt::Ok{};
  }

  static adt::Result<ValueT> StaticMakeFusionOpCodeGenClass(
      const ValueT& self_val, const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, axpr::TryGetImpl<Self>(self_val));
    return This{}.MakeFusionOpCodeGenClass(self, args);
  }

  using NativeOrRefIrValue = ir_match::NativeOrRefIrValue<BirNode>;

  adt::Result<ValueT> MakeFusionOpCodeGenClass(
      const Self& self, const std::vector<ValueT>& packed_args_vec) {
    const auto& packed_args = axpr::CastToPackedArgs(packed_args_vec);
    const auto& [args, kwargs] = *packed_args;
    ADT_CHECK(args->size() == 1) << adt::errors::TypeError{
        "'CodeGenCtx.conver_fusion_op_to_function' takes 1 positional "
        "arguments but " +
        std::to_string(args->size()) + " were given."};
    ADT_LET_CONST_REF(ir_op, IrOp<BirNode>::CastFrom(args->at(0)))
        << adt::errors::TypeError{
               std::string() +
               "the positional argument 1 of "
               "'CodeGenCtx.conver_fusion_op_to_function' should "
               "be able to cast to a NativeIrOp, PackedIrOp or RefIrOp."};
    ADT_LET_CONST_REF(input_index_loop_anchor_flags_lst,
                      kwargs->template Get<adt::List<ValueT>>(
                          "input_index_loop_anchor_flags"))
        << adt::errors::TypeError{
               std::string() +
               "'CodeGenCtx.input_index_loop_anchor_flags' requires bool list "
               "typed "
               "keyword argument 'input_index_loop_anchor_flags'."};
    LoopAnchorFlags input_index_loop_anchor_flags;
    {
      input_index_loop_anchor_flags->reserve(
          input_index_loop_anchor_flags_lst->size());
      for (const auto& elt : *input_index_loop_anchor_flags_lst) {
        ADT_LET_CONST_REF(mask, elt.template TryGet<bool>())
            << adt::errors::TypeError{
                   std::string() +
                   "'CodeGenCtx.input_index_loop_anchor_flags' requires bool "
                   "list typed "
                   "keyword argument 'input_index_loop_anchor_flags'."};
        input_index_loop_anchor_flags->emplace_back(
            tLoopAnchorFlag<bool>{mask});
      }
    }
    ADT_LET_CONST_REF(output_index_loop_anchor_flags_lst,
                      kwargs->template Get<adt::List<ValueT>>(
                          "output_index_loop_anchor_flags"))
        << adt::errors::TypeError{
               std::string() +
               "'CodeGenCtx.output_index_loop_anchor_flags' requires bool list "
               "typed "
               "keyword argument 'output_index_loop_anchor_flags'."};
    LoopAnchorFlags output_index_loop_anchor_flags;
    {
      output_index_loop_anchor_flags->reserve(
          output_index_loop_anchor_flags_lst->size());
      for (const auto& elt : *output_index_loop_anchor_flags_lst) {
        ADT_LET_CONST_REF(mask, elt.template TryGet<bool>())
            << adt::errors::TypeError{
                   std::string() +
                   "'CodeGenCtx.output_index_loop_anchor_flags' requires bool "
                   "list typed "
                   "keyword argument 'output_index_loop_anchor_flags'."};
        output_index_loop_anchor_flags->emplace_back(
            tLoopAnchorFlag<bool>{mask});
      }
    }

    OpCodeGenCtx<BirNode> op_code_gen_ctx{self.shared_ptr(),
                                          input_index_loop_anchor_flags,
                                          output_index_loop_anchor_flags};
    ADT_LET_CONST_REF(
        class_attrs,
        ConvertFusionOpToClassAttrs<BirNode>(op_code_gen_ctx, ir_op));
    return axpr::TypeImpl<axpr::ClassInstance<ValueT>>(class_attrs);
  }
};

}  // namespace ap::code_gen

namespace ap::axpr {

template <typename ValueT, typename BirNode>
struct MethodClassImpl<ValueT, ap::code_gen::CodeGenCtx<BirNode>>
    : public ap::code_gen::CodeGenCtxMethodClass<ValueT, BirNode> {};

template <typename ValueT, typename BirNode>
struct MethodClassImpl<ValueT, TypeImpl<ap::code_gen::CodeGenCtx<BirNode>>>
    : public EmptyMethodClass<ValueT> {};

}  // namespace ap::axpr
