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

#include "paddle/phi/common/ap/dispatch_ctx_value.h"
#include <type_traits>
#include "paddle/pir/include/dialect/pexpr/arithmetic_type_util.h"

namespace ap::kernel_dispatch {

using pexpr::ArithmeticType;
using pexpr::ArithmeticValue;
using pexpr::BuiltinFuncType;
using pexpr::CastToArithmeticValue;
using pexpr::CastToBuiltinValue;
using pexpr::CppArithmeticType;
using pexpr::CppPointerType;
using pexpr::Method;
using pexpr::PointerType;
using pexpr::PointerValue;

template <>
Result<adt::Ok> DispatchRawContextImpl<Val>::LaunchCudaKernel(
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
                       arg_type.name() + "' to '" + defined_arg_type.name() +
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

namespace {

Result<Val> ArgValueStaticCast(const Val& self, const std::vector<Val>& args) {
  if (args.size() != 2) {
    return TypeError{std::string() + "static_cast take 2 arguments. but " +
                     std::to_string(args.size()) + " were given."};
  }
  const Result<ArithmeticType>& arg_type =
      CastToBuiltinValue<ArithmeticType>(args.at(0));
  ADT_RETURN_IF_ERROR(arg_type);
  const Result<ArithmeticValue>& arg_value =
      CastToBuiltinValue<ArithmeticValue>(args.at(1));
  ADT_RETURN_IF_ERROR(arg_value);
  const auto& arithmetic_value = pexpr::ArithmeticValueStaticCast(
      arg_type.GetOkValue(), arg_value.GetOkValue());
  ADT_RETURN_IF_ERROR(arithmetic_value);
  return arithmetic_value.GetOkValue();
}

template <typename T>
using TensorGetAttrT = Result<Val> (*)(const T& tensor, const std::string&);

template <typename T>
Result<Val> TensorShapeGetAttr(const T& tensor, const std::string&) {
  return tensor->dims;
}

template <typename T>
const T* GetConstTensorDataPtr(const pexpr::CppArithmeticType<T>&,
                               const ConstTensorData& tensor) {
  return tensor.template data<T>();
}

template <typename T>
T* GetMutableTensorDataPtr(const pexpr::CppArithmeticType<T>&,
                           const MutableTensorData& tensor) {
  return tensor.template data<T>();
}

template <typename T>
Result<Val> TensorDataGetAttr(const T& tensor, const std::string&);

template <>
Result<Val> TensorDataGetAttr(const ConstTensor<Val>& tensor,
                              const std::string&) {
  phi::DataType dtype = tensor->tensor_data.dtype();
  const auto& arithmetic_type = pexpr::GetArithmeticTypeFromPhiDataType(dtype);
  ADT_RETURN_IF_ERROR(arithmetic_type);
  return arithmetic_type.GetOkValue().Match(
      [&](const adt::Undefined&) -> Result<Val> {
        return TypeError{"dtype is invalid."};
      },
      [&](const auto& impl) -> Result<Val> {
        return PointerValue{GetConstTensorDataPtr(impl, tensor->tensor_data)};
      });
}

template <>
Result<Val> TensorDataGetAttr(const MutableTensor<Val>& tensor,
                              const std::string&) {
  phi::DataType dtype = tensor->tensor_data.dtype();
  const auto& arithmetic_type = pexpr::GetArithmeticTypeFromPhiDataType(dtype);
  ADT_RETURN_IF_ERROR(arithmetic_type);
  return arithmetic_type.GetOkValue().Match(
      [&](const adt::Undefined&) -> Result<Val> {
        return TypeError{"dtype is invalid."};
      },
      [&](const auto& impl) -> Result<Val> {
        return PointerValue{GetMutableTensorDataPtr(impl, tensor->tensor_data)};
      });
}

template <typename T>
Result<Val> TensorGetAttr(const T& tensor, const std::string& name) {
  static const std::unordered_map<std::string, TensorGetAttrT<T>> map{
      {"shape", &TensorShapeGetAttr<T>},
      {"data_ptr", &TensorDataGetAttr<T>},
  };
  const auto& iter = map.find(name);
  if (iter == map.end()) {
    return AttributeError{std::string("'Tensor' has no attribute '") + name +
                          "'"};
  }
  return iter->second(tensor, name);
}

using KernelRawCtxGettAttrT =
    Result<Val> (*)(const DispatchRawContext<Val>& raw_ctx, const std::string&);

Result<Val> DispatchRawContextGetInputs(const DispatchRawContext<Val>& raw_ctx,
                                        const std::string&) {
  return raw_ctx->inputs;
}

Result<Val> DispatchRawContextGetOutputs(const DispatchRawContext<Val>& raw_ctx,
                                         const std::string&) {
  return raw_ctx->outputs;
}

Result<Val> MakeDispatchCtx(const Val& self, const std::vector<Val>& args) {
  if (args.size() != 1) {
    return TypeError{std::string() +
                     "'DispatchRawCtx.DispatchCtx' takes 1 arguments, but " +
                     std::to_string(args.size()) + "were given."};
  }
  const Result<DispatchRawContext<Val>>& raw_ctx =
      CastToCustomValue<DispatchRawContext<Val>>(self);
  ADT_RETURN_IF_ERROR(raw_ctx);
  const Result<pexpr::Object<Val>>& object = args.at(0).Match(
      [&](const pexpr::Object<Val>& obj) -> Result<pexpr::Object<Val>> {
        return obj;
      },
      [&](const pexpr::Nothing&) -> Result<pexpr::Object<Val>> {
        return pexpr::Object<Val>{};
      },
      [&](const auto&) -> Result<pexpr::Object<Val>> {
        return TypeError{std::string() +
                         "the first argument of 'DispatchRawCtx.DispatchCtx' "
                         "must be an object."};
      });
  ADT_RETURN_IF_ERROR(object);
  return DispatchContext<Val>{raw_ctx.GetOkValue(), object.GetOkValue()};
}

Result<Val> DispatchRawContextMakeDispatchCtx(
    const DispatchRawContext<Val>& raw_ctx, const std::string&) {
  return Method<Val>{raw_ctx, Val{&MakeDispatchCtx}};
}

Result<Val> DispatchRawContextGetAttr(const DispatchRawContext<Val>& raw_ctx,
                                      const std::string& name) {
  static const std::unordered_map<std::string, KernelRawCtxGettAttrT> map{
      {"inputs", &DispatchRawContextGetInputs},
      {"outputs", &DispatchRawContextGetOutputs},
      {"DispatcherCtx", &DispatchRawContextMakeDispatchCtx},
  };
  const auto& iter = map.find(name);
  if (iter == map.end()) {
    return AttributeError{
        std::string("'DispatchRawContext' has no attribute '") + name + "'"};
  }
  return iter->second(raw_ctx, name);
}

Result<Val> DispatchContextGetInputs(const DispatchContext<Val>& ctx,
                                     const std::string& attr_name) {
  return DispatchRawContextGetInputs(ctx->raw_ctx, attr_name);
}

Result<Val> DispatchContextGetOutputs(const DispatchContext<Val>& ctx,
                                      const std::string& attr_name) {
  return DispatchRawContextGetOutputs(ctx->raw_ctx, attr_name);
}

Result<adt::List<ArgValue>> GetKernelArgs(const Val& args) {
  const Result<adt::List<Val>>& arg_list =
      CastToBuiltinValue<adt::List<Val>>(args);
  ADT_RETURN_IF_ERROR(arg_list);
  adt::List<ArgValue> ret;
  ret->reserve(arg_list.GetOkValue()->size());
  for (const auto& arg : *arg_list.GetOkValue()) {
    const Result<ArgValue>& arg_value = CastToArgValue(arg);
    ADT_RETURN_IF_ERROR(arg_value);
    ret->emplace_back(arg_value.GetOkValue());
  }
  return ret;
}

Result<Val> LaunchCuda(const Val& self, const std::vector<Val>& args) {
  if (args.size() != 4) {
    return TypeError{
        std::string() +
        "DispatchCtx.launch_cuda take 6 arguments (including self) but " +
        std::to_string(args.size()) + " were given."};
  }
  const Result<DispatchContext<Val>>& ctx =
      CastToCustomValue<DispatchContext<Val>>(self);
  ADT_RETURN_IF_ERROR(ctx);
  const Result<std::string>& func_name =
      CastToBuiltinValue<std::string>(args.at(0));
  ADT_RETURN_IF_ERROR(func_name);
  const Result<int64_t>& num_blocks =
      CastToArithmeticValue<int64_t>(args.at(1));
  ADT_RETURN_IF_ERROR(num_blocks);
  const Result<int64_t>& num_threads =
      CastToArithmeticValue<int64_t>(args.at(2));
  ADT_RETURN_IF_ERROR(num_threads);
  const Result<adt::List<ArgValue>>& kernel_args = GetKernelArgs(args.at(3));
  ADT_RETURN_IF_ERROR(kernel_args);
  const Result<adt::Ok>& ret =
      ctx.GetOkValue()->raw_ctx->LaunchCudaKernel(func_name.GetOkValue(),
                                                  num_blocks.GetOkValue(),
                                                  num_threads.GetOkValue(),
                                                  kernel_args.GetOkValue());
  ADT_RETURN_IF_ERROR(ret);
  return adt::Nothing{};
}

Result<Val> DispatchContextLaunchCuda(const DispatchContext<Val>& ctx,
                                      const std::string&) {
  return pexpr::Method<Val>{ctx, BuiltinFuncType<Val>{&LaunchCuda}};
}

template <BuiltinFuncType<Val> BuiltinFunc>
Result<Val> MakeDispatchContextMethod(const DispatchContext<Val>& ctx,
                                      const std::string&) {
  return pexpr::Method<Val>{ctx, BuiltinFuncType<Val>{BuiltinFunc}};
}

template <typename T>
Result<Val> MakeDefineCtxArithmeticType(const DispatchContext<Val>& ctx,
                                        const std::string&) {
  return ArithmeticType{CppArithmeticType<T>{}};
}

template <typename T>
Result<Val> MakeDefineCtxPointerType(const DispatchContext<Val>& ctx,
                                     const std::string&) {
  return PointerType{CppPointerType<T>{}};
}

using KernelCtxGettAttrT = Result<Val> (*)(const DispatchContext<Val>& ctx,
                                           const std::string&);

Result<Val> DispatchContextGetAttr(const DispatchContext<Val>& ctx,
                                   const std::string& name) {
  static const std::unordered_map<std::string, KernelCtxGettAttrT> map{
      {"static_cast", &MakeDispatchContextMethod<&ArgValueStaticCast>},
      {"inputs", &DispatchContextGetInputs},
      {"outputs", &DispatchContextGetOutputs},
      {"launch_cuda", &MakeDispatchContextMethod<&LaunchCuda>},
#define MAKE_CPP_TYPE_CASE(cpp_type, enum_type)                     \
  {#cpp_type, &MakeDefineCtxArithmeticType<cpp_type>},              \
      {"const_" #cpp_type, &MakeDefineCtxArithmeticType<cpp_type>}, \
      {#cpp_type "_ptr", &MakeDefineCtxPointerType<cpp_type*>},     \
      {"const_" #cpp_type "_ptr", &MakeDefineCtxPointerType<const cpp_type*>},
      PD_FOR_EACH_DATA_TYPE(MAKE_CPP_TYPE_CASE)
#undef MAKE_CPP_TYPE_CASE
#define MAKE_INT_CPP_TYPE_CASE(cpp_type)                                \
  {#cpp_type, &MakeDefineCtxArithmeticType<cpp_type##_t>},              \
      {"const_" #cpp_type, &MakeDefineCtxArithmeticType<cpp_type##_t>}, \
      {#cpp_type "_ptr", &MakeDefineCtxPointerType<cpp_type##_t*>},     \
      {"const_" #cpp_type "_ptr",                                       \
       &MakeDefineCtxPointerType<const cpp_type##_t*>},
          AP_FOR_EACH_INT_TYPE(MAKE_INT_CPP_TYPE_CASE)
#undef MAKE_INT_CPP_TYPE_CASE
              {"void_ptr", &MakeDefineCtxPointerType<void*>},
      {"const_void_ptr", &MakeDefineCtxPointerType<const void*>},
  };
  const auto& iter = map.find(name);
  if (iter == map.end()) {
    return AttributeError{std::string("'DispatchContext' has no attribute '") +
                          name + "'"};
  }
  return iter->second(ctx, name);
}

}  // namespace

Result<Val> CustomGetAttr(const CustomValue& custom_value,
                          const std::string& name) {
  return custom_value.Match(
      [&](const ConstTensor<Val>& tensor) -> Result<Val> {
        return TensorGetAttr(tensor, name);
      },
      [&](const MutableTensor<Val>& tensor) -> Result<Val> {
        return TensorGetAttr(tensor, name);
      },
      [&](const DispatchRawContext<Val>& raw_ctx) -> Result<Val> {
        return DispatchRawContextGetAttr(raw_ctx, name);
      },
      [&](const DispatchContext<Val>& raw_ctx) -> Result<Val> {
        return DispatchContextGetAttr(raw_ctx, name);
      });
}

}  // namespace ap::kernel_dispatch
