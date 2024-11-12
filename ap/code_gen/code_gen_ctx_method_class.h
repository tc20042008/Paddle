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
#include "ap/code_gen/kernel_arg_id_ordered_set.h"
#include "ap/code_gen/op_code_gen_ctx.h"
#include "ap/code_module/module.h"
#include "ap/index_expr/index_tuple_expr.h"
#include "ap/ir_match/native_or_ref_ir_value.h"
#include "ap/registry/registry_singleton.h"

namespace ap::code_gen {

using ap::axpr::BuiltinBinaryFuncT;
using ap::axpr::BuiltinFuncType;
using ap::axpr::BuiltinUnaryFuncT;
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
    if (attr_name == "op_code_gen") {
      return axpr::Method<ValueT>{self, &This::StaticOpCodeGen};
    }
    if (attr_name == "render_module_template") {
      return axpr::Method<ValueT>{self, &This::StaticRenderModuleTemplate};
    }
    if (attr_name == "make_kernel_args_getter") {
      return axpr::Method<ValueT>{self, &This::StaticMakeKernelArgsGetter};
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
    return OutTensorDataPtrKernelArgId<BirNode>{ir_value};
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
    return InTensorDataPtrKernelArgId<BirNode>{ir_value};
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
    return DimExprKernelArgId<BirNode>{dim_expr};
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

  static adt::Result<ValueT> StaticMakeKernelArgsGetter(
      const ValueT& self_val, const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template TryGet<Self>());
    return This{}.MakeKernelArgsGetter(self, args);
  }

  adt::Result<ValueT> MakeKernelArgsGetter(const Self& self,
                                           const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() +
        "make_kernel_args_getter() method takes 1 argument but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(ordered_set,
                      args.at(0).template TryGet<axpr::OrderedSet<ValueT>>())
        << adt::errors::TypeError{std::string() +
                                  "the argument 1 of make_kernel_args_getter() "
                                  "should be OrderedSet, " +
                                  axpr::GetTypeName(args.at(0)) + " found."};
    ADT_LET_CONST_REF(kernel_arg_ids, GetKernelArgIds(ordered_set));
    return MakeKernelArgsGetterByKernelArgIds(self, kernel_arg_ids);
  }

  adt::Result<std::list<KernelArgId<BirNode>>> GetKernelArgIds(
      const axpr::OrderedSet<ValueT>& ordered_set) {
    std::list<KernelArgId<BirNode>> kernel_arg_ids{};
    int i = 0;
    for (const auto& elt : ordered_set->items()) {
      ADT_LET_CONST_REF(kernel_arg_id, KernelArgId<BirNode>::CastFrom(elt))
          << adt::errors::TypeError{std::string() + "sequence item " +
                                    std::to_string(i) +
                                    ": expected KernelArgId, " +
                                    axpr::GetTypeName(elt) + " found."};
      kernel_arg_ids.emplace_back(kernel_arg_id);
      ++i;
    }
    return kernel_arg_ids;
  }

  adt::Result<std::unordered_map<KernelArgId<BirNode>, std::string>>
  CastToKernelArgId2Name(const axpr::OrderedDict<ValueT>& ordered_dict) {
    std::unordered_map<KernelArgId<BirNode>, std::string> kernel_arg_id2name{};
    int i = 0;
    for (const auto& [k, v] : ordered_dict->items()) {
      ADT_LET_CONST_REF(kernel_arg_id, KernelArgId<BirNode>::CastFrom(k))
          << adt::errors::TypeError{std::string() + "pair sequence item " +
                                    std::to_string(i) +
                                    ": expected KernelArgId as pair.first, " +
                                    axpr::GetTypeName(k) + " found."};
      ADT_LET_CONST_REF(name, v.template TryGet<std::string>())
          << adt::errors::TypeError{std::string() + "pair sequence item " +
                                    std::to_string(i) +
                                    ": expected str as pair.second, " +
                                    axpr::GetTypeName(v) + " found."};
      kernel_arg_id2name[kernel_arg_id] = name;
      ++i;
    }
    return kernel_arg_id2name;
  }

  adt::Result<ValueT> MakeKernelArgsGetterByKernelArgIds(
      const Self& self, const std::list<KernelArgId<BirNode>>& kernel_arg_ids) {
    ArgSourceHelper<BirNode> helper{self->arg_source_ctx};
    ADT_LET_CONST_REF(getter,
                      helper.MakeRuntimeKerneArgsGetter(kernel_arg_ids));
    return getter;
  }

  static adt::Result<ValueT> StaticOpCodeGen(const ValueT& self_val,
                                             const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, axpr::TryGetImpl<Self>(self_val));
    return This{}.OpCodeGen(self, args);
  }

  static adt::Result<ValueT> StaticRenderModuleTemplate(
      const axpr::ApplyT<ValueT>& Apply,
      const ValueT& self_val,
      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, axpr::TryGetImpl<Self>(self_val));
    return This{}.RenderModuleTemplate(Apply, self, args);
  }

  using NativeOrRefIrValue = ir_match::NativeOrRefIrValue<BirNode>;

  adt::Result<ValueT> OpCodeGen(const Self& self,
                                const std::vector<ValueT>& packed_args_vec) {
    const auto& packed_args = axpr::CastToPackedArgs(packed_args_vec);
    const auto& [args, kwargs] = *packed_args;
    ADT_CHECK(args->size() == 1) << adt::errors::TypeError{
        "'CodeGenCtx.op_code_gen' takes 1 positional arguments but " +
        std::to_string(args->size()) + " were given."};
    ADT_LET_CONST_REF(ir_op, IrOp<BirNode>::CastFrom(args->at(0)))
        << adt::errors::TypeError{
               std::string() +
               "the positional argument 1 of 'CodeGenCtx.op_code_gen' should "
               "be able to cast to a NativeIrOp, PackedIrOp or RefIrOp."};
    ADT_LET_CONST_REF(kernel_arg_id2arg_name_val,
                      kwargs->template Get<axpr::OrderedDict<ValueT>>(
                          "kernel_arg_id_to_arg_name"))
        << adt::errors::TypeError{
               std::string() +
               "op_code_gen() requires 'OrderedDict' typed keyword argument "
               "'kernel_arg_id_to_arg_name'."};
    ADT_LET_CONST_REF(kernel_arg_id2arg_name,
                      CastToKernelArgId2Name(kernel_arg_id2arg_name_val));
    ADT_LET_CONST_REF(loop_index_tuple_expr,
                      kwargs->template Get<index_expr::IndexTupleExpr>(
                          "loop_index_tuple_expr"))
        << adt::errors::TypeError{
               std::string() +
               "'CodeGenCtx.op_code_gen' requires 'IndexTupleExpr' typed "
               "keyword argument 'loop_index_tuple_expr'."};
    std::vector<std::string> loop_var_names{};
    {
      ADT_LET_CONST_REF(
          loop_var_names_val,
          kwargs->template Get<adt::List<ValueT>>("loop_var_names"))
          << adt::errors::TypeError{std::string() +
                                    "'CodeGenCtx.op_code_gen' requires 'list' "
                                    "typed keyword argument 'loop_var_names'."};
      loop_var_names.reserve(loop_var_names_val->size());
      for (const auto& elt : *loop_var_names_val) {
        ADT_LET_CONST_REF(loop_var_name, axpr::TryGetImpl<std::string>(elt))
            << adt::errors::TypeError{
                   std::string() +
                   "keyword argument 'loop_var_names' of "
                   "'CodeGenCtx.op_code_gen' should be a list of string."};
        loop_var_names.emplace_back(loop_var_name);
      }
    }
    ADT_LET_CONST_REF(
        anchor_local_var_name,
        kwargs->template GetOpt<std::string>("anchor_local_var_name"))
        << adt::errors::TypeError{std::string() +
                                  "keyword argument 'anchor_local_var_name' of "
                                  "'CodeGenCtx.op_code_gen' should be a str"};
    std::vector<LocalVarBinding<BirNode>> local_var_name_bindings{};
    {
      ADT_LET_CONST_REF(ir_value_to_local_var_name,
                        kwargs->template Get<axpr::OrderedDict<ValueT>>(
                            "ir_value_to_local_var_name"))
          << adt::errors::TypeError{
                 std::string() +
                 "'CodeGenCtx.op_code_gen' requires 'list' typed keyword "
                 "argument 'local_var_name_bindings'."};
      local_var_name_bindings.reserve(
          ir_value_to_local_var_name->items().size());
      for (const auto& [k, v] : ir_value_to_local_var_name->items()) {
        ADT_LET_CONST_REF(local_var_name, axpr::TryGetImpl<std::string>(v))
            << adt::errors::TypeError{
                   std::string() +
                   "keyword argument 'ir_value_to_local_var_name' of "
                   "'CodeGenCtx.op_code_gen' should be OrderedDict[str, "
                   "'NativeIrValue|RefIrValue']."};
        ADT_LET_CONST_REF(ir_tensor, NativeOrRefIrValue::CastFrom(k))
            << adt::errors::TypeError{
                   std::string() +
                   "keyword argument 'ir_value_to_local_var_name' of "
                   "'CodeGenCtx.op_code_gen' should be OrderedDict[str, "
                   "'NativeIrValue|RefIrValue']."};
        LocalVarBinding<BirNode> binding{local_var_name, ir_tensor};
        local_var_name_bindings.emplace_back(binding);
      }
    }
    OpCodeGenCtx<BirNode> op_code_gen_ctx{
        self.shared_ptr(),
        loop_index_tuple_expr,
        loop_var_names,
        local_var_name_bindings,
        anchor_local_var_name,
        kernel_arg_id2arg_name,
    };
    ADT_LET_CONST_REF(code_str, OpCudaCodeGen<BirNode>(op_code_gen_ctx, ir_op));
    return code_str;
  }

  using Lambda = axpr::Lambda<axpr::CoreExpr>;
  using Object = axpr::Object<ValueT>;

  adt::Result<ValueT> RenderModuleTemplate(
      const axpr::ApplyT<ValueT>& Apply,
      const Self& self,
      const std::vector<ValueT>& packed_args_vec) {
    const auto& packed_args = axpr::CastToPackedArgs(packed_args_vec);
    const auto& [args, kwargs] = *packed_args;
    ADT_CHECK(args->size() == 1) << adt::errors::TypeError{
        std::string() +
        "'CodeGenCtx.render_module_template' takes 1 postional argument but " +
        std::to_string(args->size()) + " were given."};
    ADT_LET_CONST_REF(template_name, args->at(0).template TryGet<std::string>())
        << adt::errors::TypeError{
               std::string() +
               "the positional argument 1 of "
               "'CodeGenCtx.render_module_template' should be a 'str' but '" +
               axpr::GetTypeName(args->at(0)) + "' were given."};
    ADT_LET_CONST_REF(lambda, GetHighPriorModuleTemplate(template_name));
    ADT_LET_CONST_REF(m, CreateModule(Apply, lambda, kwargs));
    return m;
  }

  adt::Result<code_module::Module> CreateModule(
      const axpr::ApplyT<ValueT>& Apply,
      const Lambda& lambda,
      const Object& ctx) {
    ADT_LET_CONST_REF(module_val, Apply(lambda, {ctx}));
    ADT_LET_CONST_REF(m, module_val.template TryGet<code_module::Module>());
    return m;
  }

  adt::Result<Lambda> GetHighPriorModuleTemplate(
      const std::string& template_name) {
    ADT_LET_CONST_REF(registry, registry::RegistrySingleton::Singleton());
    const auto& module_templates = registry->module_template_registry_items;
    const auto& iter = module_templates.find(template_name);
    ADT_CHECK(iter != module_templates.end())
        << adt::errors::KeyError{std::string() + "no module template named '" +
                                 template_name + "' were found."};
    for (const auto& [nice, templates] : iter->second) {
      for (const auto& item : templates) {
        const auto& module_template = item->lambda->data;
        ADT_CHECK(module_template.has_value());
        return module_template.value();
      }
    }
    return adt::errors::KeyError{std::string() + "no module template named '" +
                                 template_name + "' were found."};
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
