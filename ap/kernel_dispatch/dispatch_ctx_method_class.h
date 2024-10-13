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

#include "ap/axpr/data_type_util.h"
#include "ap/axpr/method_class.h"
#include "ap/axpr/value_method_class.h"
#include "ap/kernel_dispatch/dispatch_ctx.h"
#include "ap/kernel_dispatch/dispatch_raw_ctx_method_class.h"

namespace ap::kernel_dispatch {

using ap::axpr::BuiltinBinaryFuncT;
using ap::axpr::BuiltinFuncType;
using ap::axpr::BuiltinUnaryFuncT;
using ap::axpr::CppDataType;
using ap::axpr::CppPointerType;
using ap::axpr::DataType;
using ap::axpr::DataValue;
using ap::axpr::Method;
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
  return DispatchRawCtxGetInputs(ctx->raw_ctx, attr_name);
}

template <typename Val>
Result<Val> DispatchCtxGetOutputs(const DispatchCtx<Val>& ctx,
                                  const std::string& attr_name) {
  return DispatchRawCtxGetOutputs(ctx->raw_ctx, attr_name);
}

template <typename Val>
Result<adt::List<ArgValue>> GetKernelArgs(const Val& args) {
  const Result<adt::List<Val>>& arg_list =
      MethodClass<Val>::template TryGet<adt::List<Val>>(args);
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
Result<Val> LaunchCuda(const Val& self, const std::vector<Val>& args) {
  if (args.size() != 4) {
    return TypeError{
        std::string() +
        "DispatchCtx.launch_cuda take 6 arguments (including self) but " +
        std::to_string(args.size()) + " were given."};
  }
  const Result<DispatchCtx<Val>>& ctx =
      MethodClass<Val>::template TryGet<DispatchCtx<Val>>(self);
  ADT_RETURN_IF_ERR(ctx);
  const Result<std::string>& func_name =
      MethodClass<Val>::template TryGet<std::string>(args.at(0));
  ADT_RETURN_IF_ERR(func_name);
  const Result<int64_t>& num_blocks =
      MethodClass<Val>::template TryGet<int64_t>(args.at(1));
  ADT_RETURN_IF_ERR(num_blocks);
  const Result<int64_t>& num_threads =
      MethodClass<Val>::template TryGet<int64_t>(args.at(2));
  ADT_RETURN_IF_ERR(num_threads);
  const Result<adt::List<ArgValue>>& kernel_args = GetKernelArgs(args.at(3));
  ADT_RETURN_IF_ERR(kernel_args);
  const Result<adt::Ok>& ret =
      ctx.GetOkValue()->raw_ctx->LaunchCudaKernel(func_name.GetOkValue(),
                                                  num_blocks.GetOkValue(),
                                                  num_threads.GetOkValue(),
                                                  kernel_args.GetOkValue());
  ADT_RETURN_IF_ERR(ret);
  return adt::Nothing{};
}

template <typename Val>
Result<Val> DispatchCtxLaunchCuda(const DispatchCtx<Val>& ctx,
                                  const std::string&) {
  return ap::axpr::Method<Val>{ctx, BuiltinFuncType<Val>{&LaunchCuda}};
}

template <typename Val, BuiltinFuncType<Val> BuiltinFunc>
Result<Val> MakeDispatchCtxMethod(const DispatchCtx<Val>& ctx,
                                  const std::string&) {
  return ap::axpr::Method<Val>{ctx, BuiltinFuncType<Val>{BuiltinFunc}};
}

template <typename Val, typename T>
Result<Val> MakeDefineCtxDataType(const DispatchCtx<Val>& ctx,
                                  const std::string&) {
  return DataType{CppDataType<T>{}};
}

template <typename Val, typename T>
Result<Val> MakeDefineCtxPointerType(const DispatchCtx<Val>& ctx,
                                     const std::string&) {
  return PointerType{CppPointerType<T>{}};
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
      {"launch_cuda", &MakeDispatchCtxMethod<Val, &LaunchCuda<Val>>},
#define MAKE_CPP_TYPE_CASE(cpp_type, enum_type)                      \
  {#cpp_type, &MakeDefineCtxDataType<Val, cpp_type>},                \
      {"const_" #cpp_type, &MakeDefineCtxDataType<Val, cpp_type>},   \
      {#cpp_type "_ptr", &MakeDefineCtxPointerType<Val, cpp_type*>}, \
      {"const_" #cpp_type "_ptr",                                    \
       &MakeDefineCtxPointerType<Val, const cpp_type*>},
      PD_FOR_EACH_DATA_TYPE(MAKE_CPP_TYPE_CASE)
#undef MAKE_CPP_TYPE_CASE
#define MAKE_INT_CPP_TYPE_CASE(cpp_type)                                 \
  {#cpp_type, &MakeDefineCtxDataType<Val, cpp_type##_t>},                \
      {"const_" #cpp_type, &MakeDefineCtxDataType<Val, cpp_type##_t>},   \
      {#cpp_type "_ptr", &MakeDefineCtxPointerType<Val, cpp_type##_t*>}, \
      {"const_" #cpp_type "_ptr",                                        \
       &MakeDefineCtxPointerType<Val, const cpp_type##_t*>},
          AP_FOR_EACH_INT_TYPE(MAKE_INT_CPP_TYPE_CASE)
#undef MAKE_INT_CPP_TYPE_CASE
              {"void_ptr", &MakeDefineCtxPointerType<Val, void*>},
      {"const_void_ptr", &MakeDefineCtxPointerType<Val, const void*>},
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
  using Self = DispatchCtxMethodClass;

  template <typename BuiltinUnarySymbol>
  static std::optional<BuiltinUnaryFuncT<ValueT>> GetBuiltinUnaryFunc() {
    return std::nullopt;
  }

  template <typename BultinBinarySymbol>
  static std::optional<BuiltinBinaryFuncT<ValueT>> GetBuiltinBinaryFunc() {
    if constexpr (std::is_same_v<BultinBinarySymbol,
                                 ap::axpr::builtin_symbol::GetAttr>) {
      return &Self::GetAttr;
    }
    return std::nullopt;
  }

  static adt::Result<ValueT> GetAttr(const ValueT& obj_val,
                                     const ValueT& attr_name_val) {
    const auto& opt_obj =
        MethodClass<ValueT>::template TryGet<DispatchCtx<ValueT>>(obj_val);
    ADT_RETURN_IF_ERR(opt_obj);
    const auto& obj = opt_obj.GetOkValue();
    const auto& opt_attr_name =
        MethodClass<ValueT>::template TryGet<std::string>(attr_name_val);
    ADT_RETURN_IF_ERR(opt_attr_name);
    const auto& attr_name = opt_attr_name.GetOkValue();
    return detail::DispatchCtxGetAttr<Val>(obj, attr_name);
  }
};

}  // namespace ap::kernel_dispatch

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, ap::kernel_dispatch::DispatchCtx<ValueT>> {
  using method_class = ap::kernel_dispatch::DispatchCtxMethodClass<ValueT>;

  template <typename BuiltinUnarySymbol>
  static std::optional<BuiltinUnaryFuncT<ValueT>> GetBuiltinUnaryFunc() {
    return method_class::template GetBuiltinUnaryFunc<BuiltinUnarySymbol>();
  }

  template <typename BultinBinarySymbol>
  static std::optional<BuiltinBinaryFuncT<ValueT>> GetBuiltinBinaryFunc() {
    return method_class::template GetBuiltinBinaryFunc<BultinBinarySymbol>();
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT,
                       TypeImpl<ap::kernel_dispatch::DispatchCtx<ValueT>>>
    : public EmptyMethodClass<ValueT> {};

}  // namespace ap::axpr
