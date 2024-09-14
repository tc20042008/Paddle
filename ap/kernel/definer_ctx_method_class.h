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
#include "ap/kernel/module.h"

namespace ap::kernel_define {

using ap::axpr::BuiltinBinaryFuncT;
using ap::axpr::BuiltinFuncType;
using ap::axpr::BuiltinUnaryFuncT;
using ap::axpr::CppDataType;
using ap::axpr::CppPointerType;
using ap::axpr::DataType;
using ap::axpr::MethodClass;
using ap::axpr::PointerType;

namespace detail {

template <typename Val>
Result<Val> DefinerCtxMakeModule(const Val& self,
                                 const std::vector<Val>& args) {
  if (args.size() != 2) {
    return TypeError{std::string("Definectx.module takes 2 arguments but ") +
                     std::to_string(args.size()) + "were given."};
  }
  const auto& list = args.at(0).Match(
      [&](const adt::List<Val>& l) -> adt::List<Val> { return l; },
      [&](const auto& impl) -> adt::List<Val> {
        return adt::List<Val>{Val{impl}};
      });
  adt::List<FuncDeclare> func_declares;
  func_declares->reserve(list->size());
  for (const auto& elt : *list) {
    if (!elt.template Has<FuncDeclare>()) {
      return TypeError{
          std::string("the first argument of Definectx.module must be a "
                      "func_declare object or a list of func_declare object.")};
    }
    func_declares->emplace_back(elt.template Get<FuncDeclare>());
  }
  if (!args.at(1).template Has<SourceCode>()) {
    return TypeError{
        std::string("the seoncd argument of Definectx.module must be a "
                    "source_code object.")};
  }
  const auto& source_code = args.at(1).template Get<SourceCode>();
  return Module{func_declares, source_code};
}

template <typename Val>
Result<adt::List<ArgType>> GetFuncArgTypes(const Val& val) {
  const auto& list = MethodClass<Val>::template TryGet<adt::List<Val>>(val);
  ADT_RETURN_IF_ERROR(list);
  adt::List<ArgType> ret;
  ret->reserve(list.GetOkValue()->size());
  for (const auto& elt : *list.GetOkValue()) {
    const auto& arg_type = CastToArgType(elt);
    ADT_RETURN_IF_ERROR(arg_type);
    ret->emplace_back(arg_type.GetOkValue());
  }
  return ret;
}

template <typename Val>
Result<Val> DefinerCtxMakeDeclareFunc(const Val& self,
                                      const std::vector<Val>& args) {
  if (args.size() != 2) {
    return TypeError{
        std::string("Definectx.declare_func takes 2 arguments but ") +
        std::to_string(args.size()) + "were given."};
  }
  const Result<FuncId>& func_id =
      MethodClass<Val>::template TryGet<std::string>(args.at(0));
  ADT_RETURN_IF_ERROR(func_id);
  const Result<adt::List<ArgType>>& arg_types = GetFuncArgTypes(args.at(1));
  ADT_RETURN_IF_ERROR(arg_types);
  return FuncDeclare{func_id.GetOkValue(), arg_types.GetOkValue()};
}

template <typename Val>
Result<Val> DefinerCtxMakeSource(const Val& self,
                                 const std::vector<Val>& args) {
  if (args.size() != 1) {
    return TypeError{
        std::string("Definectx.declare_func takes 1 arguments. but ") +
        std::to_string(args.size()) + "were given."};
  }
  if (!args.at(0).template Has<std::string>()) {
    return TypeError{std::string(
        "the first argument of Definectx.source_code must be string.")};
  }
  return SourceCode{args.at(0).template Get<std::string>()};
}

template <typename Val, typename T>
Result<Val> DefinerCtxMakeDataType(const DefinerCtx<Val>& ctx,
                                   const std::string&) {
  return DataType{CppDataType<T>{}};
}

template <typename Val, typename T>
Result<Val> DefinerCtxMakePointerType(const DefinerCtx<Val>& ctx,
                                      const std::string&) {
  return PointerType{CppPointerType<T>{}};
}

template <typename Val, BuiltinFuncType<Val> BuiltinFunc>
Result<Val> DefinerCtxMethod(const DefinerCtx<Val>& ctx, const std::string&) {
  return ap::axpr::Method<Val>{ctx, BuiltinFunc};
}

template <typename Val>
using DefinerCtxGetAttrT = Result<Val> (*)(const DefinerCtx<Val>& ctx,
                                           const std::string&);

template <typename Val>
Result<Val> DefinerCtxGetAttr(const DefinerCtx<Val>& ctx,
                              const std::string& name) {
  static const std::unordered_map<std::string, DefinerCtxGetAttrT<Val>> map{
      {"module", &DefinerCtxMethod<Val, &DefinerCtxMakeModule<Val>>},
      {"declare_func", &DefinerCtxMethod<Val, &DefinerCtxMakeDeclareFunc<Val>>},
      {"source_code", &DefinerCtxMethod<Val, &DefinerCtxMakeSource<Val>>},
#define MAKE_CPP_TYPE_CASE(cpp_type, enum_type)                       \
  {#cpp_type, &DefinerCtxMakeDataType<Val, cpp_type>},                \
      {"const_" #cpp_type, &DefinerCtxMakeDataType<Val, cpp_type>},   \
      {#cpp_type "_ptr", &DefinerCtxMakePointerType<Val, cpp_type*>}, \
      {"const_" #cpp_type "_ptr",                                     \
       &DefinerCtxMakePointerType<Val, const cpp_type*>},
      PD_FOR_EACH_DATA_TYPE(MAKE_CPP_TYPE_CASE)
#undef MAKE_CPP_TYPE_CASE
#define MAKE_INT_CPP_TYPE_CASE(cpp_type)                                  \
  {#cpp_type, &DefinerCtxMakeDataType<Val, cpp_type##_t>},                \
      {"const_" #cpp_type, &DefinerCtxMakeDataType<Val, cpp_type##_t>},   \
      {#cpp_type "_ptr", &DefinerCtxMakePointerType<Val, cpp_type##_t*>}, \
      {"const_" #cpp_type "_ptr",                                         \
       &DefinerCtxMakePointerType<Val, const cpp_type##_t*>},
          AP_FOR_EACH_INT_TYPE(MAKE_INT_CPP_TYPE_CASE)
#undef MAKE_INT_CPP_TYPE_CASE
              {"void_ptr", &DefinerCtxMakePointerType<Val, void*>},
      {"const_void_ptr", &DefinerCtxMakePointerType<Val, const void*>},
  };
  const auto iter = map.find(name);
  if (iter == map.end()) {
    return AttributeError{
        std::string("'DefinerCtx' object has no attribute '") + name + "' "};
  }
  return iter->second(ctx, name);
}

}  // namespace detail

template <typename ValueT>
struct DefinerCtxMethodClass {
  using Self = DefinerCtxMethodClass;

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

  static adt::Result<ValueT> GetAttr(const ValueT& obj,
                                     const ValueT& attr_name_val) {
    const auto& opt_ctx =
        MethodClass<ValueT>::template TryGet<DefinerCtx<Val>>(obj);
    ADT_RETURN_IF_ERROR(opt_ctx);
    const auto& ctx = opt_ctx.GetOkValue();
    const auto& opt_attr_name =
        MethodClass<ValueT>::template TryGet<std::string>(attr_name_val);
    ADT_RETURN_IF_ERROR(opt_attr_name);
    const auto& attr_name = opt_attr_name.GetOkValue();
    return detail::DefinerCtxGetAttr<Val>(ctx, attr_name);
  }
};

}  // namespace ap::kernel_define

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, ap::kernel_define::DefinerCtx<ValueT>> {
  using method_class = ap::kernel_define::DefinerCtxMethodClass<ValueT>;

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
struct MethodClassImpl<ValueT, TypeImpl<ap::kernel_define::DefinerCtx<ValueT>>>
    : public EmptyMethodClass<ValueT> {};

}  // namespace ap::axpr
