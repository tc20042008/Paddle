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

namespace ap::kernel_dispatch {

using ap::kernel_define::CppArgType;
using pexpr::BuiltinFuncType;
using pexpr::CastToBuiltinValue;
using pexpr::InterpretFuncType;
using pexpr::MethodClosure;

template <>
Result<adt::Ok> DispatchRawContextImpl<Val>::LaunchCudaKernel(
    const std::string& func_name,
    int64_t num_blocks,
    int64_t num_threads,
    const adt::List<CppValue>& kernel_args) const {
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
    const auto& arg_type = GetArgType(kernel_arg);
    if (!(defined_arg_type.RemoveConst() == arg_type.RemoveConst())) {
      return TypeError{std::string() + "error: invalid conversion from '" +
                       arg_type.name() + "' to '" + defined_arg_type.name() +
                       "'"};
    }
    kernel_arg.Match([&](const auto& impl) {
      void_args.push_back(reinterpret_cast<void*>(
          const_cast<std::decay_t<decltype(impl)>*>(&impl)));
    });
  }
  cuda_module->LaunchCudaKernel(func_name, num_blocks, num_threads, void_args);
  return adt::Ok{};
}

namespace {

template <typename DstT, typename SrcT>
Result<Val> StaticCast(DstT arg_type, const SrcT cpp_value) {
  if constexpr (std::is_pointer_v<typename DstT::type>) {
    return InvalidArgumentError{std::string() +
                                "static_cast does not support pointers."};
  } else if constexpr (std::is_pointer_v<SrcT>) {
    return InvalidArgumentError{std::string() +
                                "static_cast does not support pointers."};
  } else if constexpr (std::is_same_v<typename DstT::type,  // NOLINT
                                      phi::dtype::pstring>) {
    return InvalidArgumentError{std::string() +
                                "static_cast does not support pstring."};
  } else if constexpr (std::is_same_v<typename DstT::type,  // NOLINT
                                      const phi::dtype::pstring>) {
    return InvalidArgumentError{std::string() +
                                "static_cast does not support pstring."};
  } else if constexpr (std::is_same_v<SrcT, phi::dtype::pstring>) {
    return InvalidArgumentError{std::string() +
                                "static_cast does not support pstring."};
  } else if constexpr (std::is_same_v<SrcT, const phi::dtype::pstring>) {
    return InvalidArgumentError{std::string() +
                                "static_cast does not support pstring."};
  } else {
    return Val{CppValue{static_cast<typename DstT::type>(cpp_value)}};
  }
}

Result<Val> CppValueStaticCast(const InterpretFuncType<Val>& Interpret,
                               const std::vector<Val>& args) {
  if (args.size() != 2) {
    return TypeError{
        std::string() +
        "CppValue.static_cast take 2 arguments (including self) but " +
        std::to_string(args.size()) + " were given."};
  }
  const Result<CppValue>& cpp_value = CastToCustomValue<CppValue>(args.at(0));
  ADT_RETURN_IF_ERROR(cpp_value);
  const Result<ArgType>& arg_type = CastToCustomValue<ArgType>(args.at(1));
  ADT_RETURN_IF_ERROR(arg_type);
  const auto& pattern_match = ::common::Overloaded{
      [&](auto arg_type_impl, auto cpp_value_impl) -> Result<Val> {
        return StaticCast(arg_type_impl, cpp_value_impl);
      }};
  return std::visit(pattern_match,
                    arg_type.GetOkValue().variant(),
                    cpp_value.GetOkValue().variant());
}

Result<Val> CppValueGetAttr(const CppValue& cpp_value,
                            const std::string& name) {
  static const std::unordered_map<std::string, BuiltinFuncType<Val>> map{
      {"static_cast", &CppValueStaticCast},
  };
  const auto& iter = map.find(name);
  if (iter == map.end()) {
    return AttributeError{std::string("'CppValue' has no attribute '") + name +
                          "'"};
  }
  return MethodClosure<Val>{Val{cpp_value}, Val{iter->second}};
}

template <typename T>
using TensorGetAttrT = Result<Val> (*)(const T& tensor, const std::string&);

template <typename T>
Result<Val> TensorShapeGetAttr(const T& tensor, const std::string&) {
  return tensor->dims;
}

template <typename T>
const T* GetConstTensorDataPtr(const CppArgType<T>&,
                               const ConstTensorData& tensor) {
  return tensor.template data<T>();
}

template <typename T>
T* GetMutableTensorDataPtr(const CppArgType<T>&,
                           const MutableTensorData& tensor) {
  return tensor.template data<T>();
}

template <typename T>
Result<Val> TensorDataGetAttr(const T& tensor, const std::string&);

template <>
Result<Val> TensorDataGetAttr(const ConstTensor<Val>& tensor,
                              const std::string&) {
  phi::DataType dtype = tensor->tensor_data.dtype();
  const auto& arg_type = kernel_define::ArgType::MakeFromPhiDataType(dtype);
  ADT_RETURN_IF_ERROR(arg_type);
  return arg_type.GetOkValue().Match([&](const auto& impl) -> Val {
    return CppValue{GetConstTensorDataPtr(impl, tensor->tensor_data)};
  });
}

template <>
Result<Val> TensorDataGetAttr(const MutableTensor<Val>& tensor,
                              const std::string&) {
  phi::DataType dtype = tensor->tensor_data.dtype();
  const auto& arg_type = kernel_define::ArgType::MakeFromPhiDataType(dtype);
  ADT_RETURN_IF_ERROR(arg_type);
  return arg_type.GetOkValue().Match([&](const auto& impl) -> Val {
    return CppValue{GetMutableTensorDataPtr(impl, tensor->tensor_data)};
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

Result<Val> MakeDispatchCtx(const InterpretFuncType<Val>& Interpret,
                            const std::vector<Val>& args) {
  if (args.size() != 2) {
    return TypeError{std::string() +
                     "'DispatchRawCtx.DispatchCtx' takes 2 arguments, but " +
                     std::to_string(args.size()) + "were given."};
  }
  const Result<DispatchRawContext<Val>>& raw_ctx = args.at(0).Match(
      [&](const DispatchRawContext<Val>& raw_ctx)
          -> Result<DispatchRawContext<Val>> { return raw_ctx; },
      [&](const auto&) -> Result<DispatchRawContext<Val>> {
        return TypeError{std::string() +
                         "the self argument of 'DispatchRawCtx.DispatchCtx' "
                         "must be an DispatchRawContext."};
      });
  ADT_RETURN_IF_ERROR(raw_ctx);
  const Result<pexpr::Object<Val>>& object = args.at(1).Match(
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
  return MethodClosure<Val>{raw_ctx,
                            Val{BuiltinFuncType<Val>(&MakeDispatchCtx)}};
}

Result<Val> DispatchRawContextGetAttr(const DispatchRawContext<Val>& raw_ctx,
                                      const std::string& name) {
  static const std::unordered_map<std::string, KernelRawCtxGettAttrT> map{
      {"inputs", &DispatchRawContextGetInputs},
      {"outputs", &DispatchRawContextGetOutputs},
      {"DispatchCtx", &DispatchRawContextMakeDispatchCtx},
  };
  const auto& iter = map.find(name);
  if (iter == map.end()) {
    return AttributeError{
        std::string("'DispatchRawContext' has no attribute '") + name + "'"};
  }
  return iter->second(raw_ctx, name);
}

using KernelCtxGettAttrT = Result<Val> (*)(const DispatchContext<Val>& ctx,
                                           const std::string&);

Result<Val> DispatchContextGetInputs(const DispatchContext<Val>& ctx,
                                     const std::string& attr_name) {
  return DispatchRawContextGetInputs(ctx->raw_ctx, attr_name);
}

Result<Val> DispatchContextGetOutputs(const DispatchContext<Val>& ctx,
                                      const std::string& attr_name) {
  return DispatchRawContextGetOutputs(ctx->raw_ctx, attr_name);
}

Result<adt::List<CppValue>> GetKernelArgs(const Val& args) {
  const Result<adt::List<Val>>& arg_list =
      CastToBuiltinValue<adt::List<Val>>(args);
  ADT_RETURN_IF_ERROR(arg_list);
  adt::List<CppValue> ret;
  ret->reserve(arg_list.GetOkValue()->size());
  for (const auto& arg : *arg_list.GetOkValue()) {
    const Result<CppValue>& cpp_arg = CastToCustomValue<CppValue>(arg);
    ADT_RETURN_IF_ERROR(cpp_arg);
    ret->emplace_back(cpp_arg.GetOkValue());
  }
  return ret;
}

Result<Val> LaunchCuda(const InterpretFuncType<Val>& Interpret,
                       const std::vector<Val>& args) {
  if (args.size() != 5) {
    return TypeError{
        std::string() +
        "DispatchCtx.cuda_call take 6 arguments (including self) but " +
        std::to_string(args.size()) + " were given."};
  }
  const Result<DispatchContext<Val>>& ctx =
      CastToCustomValue<DispatchContext<Val>>(args.at(0));
  ADT_RETURN_IF_ERROR(ctx);
  const Result<std::string>& func_name =
      CastToBuiltinValue<std::string>(args.at(1));
  ADT_RETURN_IF_ERROR(func_name);
  const Result<int64_t>& num_blocks = CastToBuiltinValue<int64_t>(args.at(2));
  ADT_RETURN_IF_ERROR(num_blocks);
  const Result<int64_t>& num_threads = CastToBuiltinValue<int64_t>(args.at(3));
  ADT_RETURN_IF_ERROR(num_threads);
  const Result<adt::List<CppValue>>& kernel_args = GetKernelArgs(args.at(4));
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
  return pexpr::MethodClosure<Val>{ctx, BuiltinFuncType<Val>{&LaunchCuda}};
}

template <Result<Val> (*BuiltinFunc)(const InterpretFuncType<Val>&,
                                     const std::vector<Val>&)>
Result<Val> MakeDispatchContextMethod(const DispatchContext<Val>& ctx,
                                      const std::string&) {
  return pexpr::MethodClosure<Val>{ctx, BuiltinFuncType<Val>{BuiltinFunc}};
}

template <typename T>
Result<Val> MakeDefineCtxArgType(const InterpretFuncType<Val>& Interpret,
                                 const std::vector<Val>& args) {
  return ArgType{CppArgType<T>{}};
}

Result<Val> DispatchContextGetAttr(const DispatchContext<Val>& ctx,
                                   const std::string& name) {
  static const std::unordered_map<std::string, KernelCtxGettAttrT> map{
      {"inputs", &DispatchContextGetInputs},
      {"outputs", &DispatchContextGetOutputs},
      {"launch_cuda", &MakeDispatchContextMethod<&LaunchCuda>},
#define MAKE_CPP_TYPE_CASE(cpp_type, enum_type)                             \
  {#cpp_type, &MakeDispatchContextMethod<&MakeDefineCtxArgType<cpp_type>>}, \
      {"const_" #cpp_type,                                                  \
       &MakeDispatchContextMethod<&MakeDefineCtxArgType<const cpp_type>>},  \
      {#cpp_type "_ptr",                                                    \
       &MakeDispatchContextMethod<&MakeDefineCtxArgType<cpp_type*>>},       \
      {"const_" #cpp_type "_ptr",                                           \
       &MakeDispatchContextMethod<&MakeDefineCtxArgType<const cpp_type*>>},
      PD_FOR_EACH_DATA_TYPE(MAKE_CPP_TYPE_CASE)
#undef MAKE_CPP_TYPE_CASE
          {"void_ptr",
           &MakeDispatchContextMethod<&MakeDefineCtxArgType<void*>>},
      {"const_void_ptr",
       &MakeDispatchContextMethod<&MakeDefineCtxArgType<const void*>>},
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
      [&](const CppValue& cpp_value) -> Result<Val> {
        return CppValueGetAttr(cpp_value, name);
      },
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
      },
      [&](const kernel_define::ArgType&) -> Result<Val> {
        return AttributeError{
            std::string("'ArgType' object has no attribute '") + name + "' "};
      });
}

}  // namespace ap::kernel_dispatch
