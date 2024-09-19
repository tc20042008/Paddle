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
#include "ap/kernel/definer_ctx.h"
#include "ap/kernel/definer_raw_ctx.h"

namespace ap::kernel_define {

using ap::axpr::BuiltinBinaryFuncT;
using ap::axpr::BuiltinUnaryFuncT;
using ap::axpr::MethodClass;

template <typename ValueT>
struct DefinerRawCtxMethodClass {
  using Self = DefinerRawCtxMethodClass;

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
        MethodClass<ValueT>::template TryGet<DefinerRawCtx>(obj_val);
    ADT_RETURN_IF_ERR(opt_obj);
    const auto& obj = opt_obj.GetOkValue();
    const auto& opt_attr_name =
        MethodClass<ValueT>::template TryGet<std::string>(attr_name_val);
    ADT_RETURN_IF_ERR(opt_attr_name);
    const auto& attr_name = opt_attr_name.GetOkValue();
    return Self::DefinerRawCtxGetAttr(obj, attr_name);
  }

  using DefinerRawCtxGettAttrT =
      Result<ValueT> (*)(const DefinerRawCtx& raw_ctx, const std::string&);

  static adt::Result<ValueT> DefinerRawCtxGetAttr(const DefinerRawCtx& raw_ctx,
                                                  const std::string& name) {
    static const std::unordered_map<std::string, DefinerRawCtxGettAttrT> map{
        {"DefinerCtx", &Self::DefineRawContextMakeDefinerCtx},
    };
    const auto& iter = map.find(name);
    if (iter == map.end()) {
      return AttributeError{std::string("'DefinerRawCtx' has no attribute '") +
                            name + "'"};
    }
    return iter->second(raw_ctx, name);
  }

  static adt::Result<ValueT> DefineRawContextMakeDefinerCtx(
      const DefinerRawCtx& raw_ctx, const std::string&) {
    return ap::axpr::Method<ValueT>{raw_ctx, &Self::MakeDefinerCtx};
  }

  static adt::Result<ValueT> MakeDefinerCtx(const ValueT& obj,
                                            const std::vector<ValueT>& args) {
    if (args.size() != 1) {
      return adt::errors::TypeError{
          std::string() + "'DefinerRawCtx.DefinerCtx' takes 1 arguments, but " +
          std::to_string(args.size()) + "were given."};
    }

    const auto& raw_ctx =
        MethodClass<ValueT>::template TryGet<DefinerRawCtx>(obj);
    ADT_RETURN_IF_ERR(raw_ctx);
    const Result<ap::axpr::Object<ValueT>>& object = args.at(0).Match(
        [&](const ap::axpr::Object<ValueT>& obj)
            -> Result<ap::axpr::Object<ValueT>> { return obj; },
        [&](const ap::axpr::Nothing&) -> Result<ap::axpr::Object<ValueT>> {
          return ap::axpr::Object<ValueT>{};
        },
        [&](const auto&) -> Result<ap::axpr::Object<ValueT>> {
          return adt::errors::TypeError{
              std::string() +
              "the first argument of 'DefinerRawCtx.DefinerCtx' "
              "must be an object."};
        });
    ADT_RETURN_IF_ERR(object);
    return DefinerCtx<ValueT>{raw_ctx.GetOkValue(), object.GetOkValue()};
  }
};

}  // namespace ap::kernel_define

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, ap::kernel_define::DefinerRawCtx> {
  using method_class = ap::kernel_define::DefinerRawCtxMethodClass<ValueT>;

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
struct MethodClassImpl<ValueT, TypeImpl<ap::kernel_define::DefinerRawCtx>>
    : public EmptyMethodClass<ValueT> {};

}  // namespace ap::axpr
