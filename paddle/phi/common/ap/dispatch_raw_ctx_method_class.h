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

#include "paddle/phi/common/ap/dispatch_raw_ctx.h"
#include "paddle/pir/include/dialect/pexpr/data_type_util.h"
#include "paddle/pir/include/dialect/pexpr/method_class.h"

namespace ap::kernel_dispatch {

using pexpr::BuiltinFuncType;
using pexpr::CppDataType;
using pexpr::CppPointerType;
using pexpr::DataType;
using pexpr::Method;
using pexpr::MethodClass;
using pexpr::PointerType;
using pexpr::PointerValue;

namespace detail {

template <typename Val>
using KernelRawCtxGettAttrT =
    Result<Val> (*)(const DispatchRawCtx<Val>& raw_ctx, const std::string&);

template <typename Val>
Result<Val> DispatchRawCtxGetInputs(const DispatchRawCtx<Val>& raw_ctx,
                                    const std::string&) {
  return raw_ctx->inputs;
}

template <typename Val>
Result<Val> DispatchRawCtxGetOutputs(const DispatchRawCtx<Val>& raw_ctx,
                                     const std::string&) {
  return raw_ctx->outputs;
}

template <typename Val>
Result<Val> MakeDispatchCtx(const Val& self, const std::vector<Val>& args) {
  if (args.size() != 1) {
    return TypeError{std::string() +
                     "'DispatchRawCtx.DispatchCtx' takes 1 arguments, but " +
                     std::to_string(args.size()) + "were given."};
  }
  const Result<DispatchRawCtx<Val>>& raw_ctx =
      MethodClass<Val>::template TryGet<DispatchRawCtx<Val>>(self);
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
  return DispatchCtx<Val>{raw_ctx.GetOkValue(), object.GetOkValue()};
}

template <typename Val>
Result<Val> DispatchRawCtxMakeDispatchCtx(const DispatchRawCtx<Val>& raw_ctx,
                                          const std::string&) {
  return Method<Val>{raw_ctx, Val{&MakeDispatchCtx<Val>}};
}

template <typename Val>
Result<Val> DispatchRawCtxGetAttr(const DispatchRawCtx<Val>& raw_ctx,
                                  const std::string& name) {
  static const std::unordered_map<std::string, KernelRawCtxGettAttrT<Val>> map{
      {"inputs", &DispatchRawCtxGetInputs<Val>},
      {"outputs", &DispatchRawCtxGetOutputs<Val>},
      {"DispatcherCtx", &DispatchRawCtxMakeDispatchCtx<Val>},
  };
  const auto& iter = map.find(name);
  if (iter == map.end()) {
    return AttributeError{std::string("'DispatchRawCtx' has no attribute '") +
                          name + "'"};
  }
  return iter->second(raw_ctx, name);
}

}  // namespace detail

template <typename ValueT>
struct DispatchRawCtxMethodClass {
  using Self = DispatchRawCtxMethodClass;

  template <typename BuiltinUnarySymbol>
  static std::optional<BuiltinUnaryFuncT<ValueT>> GetBuiltinUnaryFunc() {
    return std::nullopt;
  }

  template <typename BultinBinarySymbol>
  static std::optional<BuiltinBinaryFuncT<ValueT>> GetBuiltinBinaryFunc() {
    if constexpr (std::is_same_v<BultinBinarySymbol,
                                 pexpr::builtin_symbol::GetAttr>) {
      return &Self::GetAttr;
    }
    return std::nullopt;
  }

  static adt::Result<ValueT> GetAttr(const ValueT& obj_val,
                                     const ValueT& attr_name_val) {
    const auto& opt_obj =
        MethodClass<ValueT>::template TryGet<DispatchRawCtx<ValueT>>(obj_val);
    ADT_RETURN_IF_ERROR(opt_obj);
    const auto& obj = opt_obj.GetOkValue();
    const auto& opt_attr_name =
        MethodClass<ValueT>::template TryGet<std::string>(attr_name_val);
    ADT_RETURN_IF_ERROR(opt_attr_name);
    const auto& attr_name = opt_attr_name.GetOkValue();
    return detail::DispatchRawCtxGetAttr<Val>(obj, attr_name);
  }
};

}  // namespace ap::kernel_dispatch

namespace pexpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, ap::kernel_dispatch::DispatchRawCtx<ValueT>> {
  using method_class = ap::kernel_dispatch::DispatchRawCtxMethodClass<ValueT>;

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
                       TypeImpl<ap::kernel_dispatch::DispatchRawCtx<ValueT>>>
    : public EmptyMethodClass<ValueT> {};

}  // namespace pexpr
