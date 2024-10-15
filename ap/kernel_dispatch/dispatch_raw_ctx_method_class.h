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
#include "ap/kernel_dispatch/dispatch_raw_ctx.h"

namespace ap::kernel_dispatch {

using ap::axpr::BuiltinFuncType;
using ap::axpr::CppDataType;
using ap::axpr::CppPointerType;
using ap::axpr::DataType;
using ap::axpr::Method;
using ap::axpr::MethodClass;
using ap::axpr::PointerType;
using ap::axpr::PointerValue;

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
Result<Val> MakeDispatchCtx(const Val& self_val, const std::vector<Val>& args) {
  ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
      std::string() + "'DispatchRawCtx.DispatchCtx' takes 1 arguments, but " +
      std::to_string(args.size()) + "were given."};
  ADT_LET_CONST_REF(raw_ctx, self_val.template TryGet<DispatchRawCtx<Val>>());
  const auto& arg0 = args.at(0);
  const auto& opt_object = arg0.Match(
      [&](const ap::axpr::Object<Val>& obj) -> Result<ap::axpr::Object<Val>> {
        return obj;
      },
      [&](const ap::axpr::Nothing&) -> Result<ap::axpr::Object<Val>> {
        return ap::axpr::Object<Val>{};
      },
      [&](const auto&) -> Result<ap::axpr::Object<Val>> {
        return TypeError{std::string() +
                         "the first argument of 'DispatchRawCtx.DispatchCtx' "
                         "must be an object. the '" +
                         axpr::GetTypeName(arg0) + "' were given."};
      });
  ADT_LET_CONST_REF(object, opt_object);
  return DispatchCtx<Val>{raw_ctx, object};
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
  using This = DispatchRawCtxMethodClass;
  using Self = DispatchRawCtx<ValueT>;

  adt::Result<ValueT> GetAttr(const Self& raw_ctx,
                              const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    return detail::DispatchRawCtxGetAttr<Val>(raw_ctx, attr_name);
  }
};

}  // namespace ap::kernel_dispatch

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, kernel_dispatch::DispatchRawCtx<ValueT>>
    : public kernel_dispatch::DispatchRawCtxMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT,
                       TypeImpl<kernel_dispatch::DispatchRawCtx<ValueT>>>
    : public EmptyMethodClass<ValueT> {};

}  // namespace ap::axpr
