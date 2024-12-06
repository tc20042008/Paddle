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

#include "paddle/ap/include/axpr/data_type_util.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/value_method_class.h"
#include "paddle/ap/include/kernel_dispatch/dispatch_ctx.h"

namespace ap::kernel_dispatch {

using ap::axpr::BuiltinBinaryFunc;
using ap::axpr::BuiltinFuncType;
using ap::axpr::BuiltinUnaryFunc;
using ap::axpr::CppDataType;
using ap::axpr::CppPointerType;
using ap::axpr::DataType;
using ap::axpr::DataValue;
using ap::axpr::MethodClass;
using ap::axpr::PointerType;
using ap::axpr::PointerValue;

template <typename Val>
Result<adt::Ok> DispatchRawCtxImpl<Val>::LaunchCudaKernel(
    const std::string& func_name,
    int64_t num_blocks,
    int64_t num_threads,
    const adt::List<ArgValue>& kernel_args) const {
  std::vector<void*> void_args;
  void_args.reserve(kernel_args->size());
  const auto& iter = this->func_name2arg_types.find(func_name);
  if (iter == this->func_name2arg_types.end()) {
    return TypeError{std::string() + "cuda kernel function '" + func_name +
                     "' not found"};
  }
  const auto& defined_arg_types = iter->second;
  if (defined_arg_types->size() != kernel_args->size()) {
    return TypeError{std::string() + "cuda kernel function '" + func_name +
                     "' takes " + std::to_string(defined_arg_types->size()) +
                     " arguments. but " + std::to_string(kernel_args->size()) +
                     " were given."};
  }
  for (int i = 0; i < defined_arg_types->size(); ++i) {
    const auto& defined_arg_type = defined_arg_types->at(i);
    const auto& kernel_arg = kernel_args->at(i);
    const auto& arg_type = kernel_arg.GetType();
    if (!(defined_arg_type == arg_type)) {
      return TypeError{std::string() + "error: invalid conversion from '" +
                       arg_type.Name() + "' to '" + defined_arg_type.Name() +
                       "'"};
    }
    kernel_arg.Match([&](const auto& impl) {
      void_args.push_back(reinterpret_cast<void*>(
          const_cast<std::decay_t<decltype(impl)>*>(&impl)));
    });
  }
  return cuda_module->LaunchCudaKernel(
      func_name, num_blocks, num_threads, void_args);
}

namespace detail {

template <typename Val>
Result<Val> DispatchCtxGetInputs(const DispatchCtx<Val>& ctx,
                                 const std::string& attr_name) {
  return ctx->raw_ctx->inputs;
}

template <typename Val>
Result<Val> DispatchCtxGetOutputs(const DispatchCtx<Val>& ctx,
                                  const std::string& attr_name) {
  return ctx->raw_ctx->outputs;
}

template <typename Val>
Result<adt::List<ArgValue>> GetKernelArgs(const Val& args) {
  const Result<adt::List<Val>>& arg_list =
      args.template TryGet<adt::List<Val>>();
  ADT_RETURN_IF_ERR(arg_list);
  adt::List<ArgValue> ret;
  ret->reserve(arg_list.GetOkValue()->size());
  for (const auto& arg : *arg_list.GetOkValue()) {
    const Result<ArgValue>& arg_value = CastToArgValue(arg);
    ADT_RETURN_IF_ERR(arg_value);
    ret->emplace_back(arg_value.GetOkValue());
  }
  return ret;
}

template <typename Val>
Result<Val> LaunchCuda(const Val& self_val, const std::vector<Val>& args) {
  ADT_CHECK(args.size() == 4) << TypeError{
      std::string() +
      "DispatchCtx.launch_cuda take 6 arguments (including self) but " +
      std::to_string(args.size()) + " were given."};
  ADT_LET_CONST_REF(ctx, axpr::Get<DispatchCtx<Val>>(self_val));
  ADT_LET_CONST_REF(func_name, args.at(0).template TryGet<std::string>());
  ADT_LET_CONST_REF(num_blocks, args.at(1).template TryGet<int64_t>());
  ADT_LET_CONST_REF(num_threads, args.at(2).template TryGet<int64_t>());
  ADT_LET_CONST_REF(kernel_args, GetKernelArgs(args.at(3)));
  ADT_RETURN_IF_ERR(ctx->raw_ctx->LaunchCudaKernel(
      func_name, num_blocks, num_threads, kernel_args));
  return adt::Nothing{};
}

template <typename Val, BuiltinFuncType<Val> BuiltinFunc>
Result<Val> MakeDispatchCtxMethod(const DispatchCtx<Val>& ctx,
                                  const std::string&) {
  return ap::axpr::Method<Val>{ctx, BuiltinFuncType<Val>{BuiltinFunc}};
}

template <typename Val, typename T>
Result<Val> DispatchCtxType(const DispatchCtx<Val>& ctx, const std::string&) {
  return ap::axpr::TypeImpl<T>{};
}

template <typename Val>
using KernelCtxGettAttrT = Result<Val> (*)(const DispatchCtx<Val>& ctx,
                                           const std::string&);

template <typename Val>
Result<Val> DispatchCtxGetAttr(const DispatchCtx<Val>& ctx,
                               const std::string& name) {
  static const std::unordered_map<std::string, KernelCtxGettAttrT<Val>> map{
      {ap::axpr::TypeImpl<ap::axpr::DataValue>{}.Name(),
       &DispatchCtxType<Val, ap::axpr::DataValue>},
      {"inputs", &DispatchCtxGetInputs<Val>},
      {"outputs", &DispatchCtxGetOutputs<Val>},
  };
  const auto& iter = map.find(name);
  if (iter == map.end()) {
    return AttributeError{std::string("'DispatchCtx' has no attribute '") +
                          name + "'"};
  }
  return iter->second(ctx, name);
}

}  // namespace detail

template <typename ValueT>
struct DispatchCtxMethodClass {
  using This = DispatchCtxMethodClass;
  using Self = DispatchCtx<ValueT>;

  static adt::Result<ValueT> GetAttr(const ValueT& self_val,
                                     const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, axpr::Get<Self>(self_val));
    ADT_CHECK(args.size() == 1);
    const auto& attr_name_val = args.at(0);
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    if (attr_name == "kernel_dispatch_const_data") {
      return self->kernel_dispatch_const_data;
    }
    return detail::DispatchCtxGetAttr<ValueT>(self, attr_name);
  }

  static adt::Result<ValueT> StaticGetInputIndexByName(
      const ValueT& self_val, const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, axpr::Get<Self>(self_val));
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() +
        "'DispatchCtx.get_input_index_by_name' takes 1 argument but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(tensor_name, args.at(0).template TryGet<std::string>())
        << adt::errors::TypeError{
               std::string() +
               "the argument 1 of 'DispatchCtx.get_input_index_by_name' should "
               "be str (not '" +
               axpr::GetTypeName(args.at(0)) + "')."};
    return This{}.GetInputIndexByName(self, tensor_name);
  }

  adt::Result<ValueT> GetInputIndexByName(const Self& self,
                                          const std::string& tensor_name) {
    const auto& data = self->kernel_dispatch_const_data;
    ADT_LET_CONST_REF(
        name2idx,
        data->template TryGet<axpr::AttrMap<axpr::SerializableValue>>(
            "__builtin_ap_kernel_input_name_to_index"));
    ADT_LET_CONST_REF(index, name2idx->template TryGet<int64_t>(tensor_name));
    return index;
  }

  static adt::Result<ValueT> StaticGetOutputIndexByName(
      const ValueT& self_val, const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, axpr::Get<Self>(self_val));
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() +
        "'DispatchCtx.get_output_index_by_name' takes 1 argument but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(tensor_name, args.at(0).template TryGet<std::string>())
        << adt::errors::TypeError{
               std::string() +
               "the argument 1 of 'DispatchCtx.get_output_index_by_name' "
               "should be str (not '" +
               axpr::GetTypeName(args.at(0)) + "')."};
    return This{}.GetOutputIndexByName(self, tensor_name);
  }

  adt::Result<ValueT> GetOutputIndexByName(const Self& self,
                                           const std::string& tensor_name) {
    const auto& data = self->kernel_dispatch_const_data;
    ADT_LET_CONST_REF(
        name2idx,
        data->template TryGet<axpr::AttrMap<axpr::SerializableValue>>(
            "__builtin_ap_kernel_output_name_to_index"));
    ADT_LET_CONST_REF(index, name2idx->template TryGet<int64_t>(tensor_name));
    return index;
  }
};

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetDispatchCtxClass() {
  using ClassT = axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>;
  using Methods = DispatchCtxMethodClass<ValueT>;
  static auto cls(
      axpr::MakeBuiltinClass<ValueT>("DispatchCtx", [&](const auto& DoEach) {
        DoEach("__getattr__", &Methods::GetAttr);
        DoEach("get_input_index_by_name", &Methods::StaticGetInputIndexByName);
        DoEach("get_output_index_by_name",
               &Methods::StaticGetOutputIndexByName);
        DoEach("launch_cuda", &detail::LaunchCuda<ValueT>);
      }));
  return ClassT(cls);
}

}  // namespace ap::kernel_dispatch
