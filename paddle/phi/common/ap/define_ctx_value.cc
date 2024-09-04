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

#include "paddle/phi/common/ap/define_ctx_value.h"
#include <unordered_map>

namespace ap::kernel_define {

using pexpr::ArithmeticType;
using pexpr::BuiltinFuncType;
using pexpr::CastToBuiltinValue;
using pexpr::CppArithmeticType;
using pexpr::CppPointerType;
using pexpr::PointerType;

namespace {

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
    if (!elt.Has<CustomValue>()) {
      return TypeError{
          std::string("the first argument of Definectx.module must be a "
                      "func_declare or a list of func_declare.")};
    }
    const auto& custom_val = elt.Get<CustomValue>();
    if (!custom_val.Has<FuncDeclare>()) {
      return TypeError{
          std::string("the first argument of Definectx.module must be a "
                      "func_declare object or a list of func_declare object.")};
    }
    func_declares->emplace_back(custom_val.Get<FuncDeclare>());
  }
  if (!args.at(1).Has<CustomValue>()) {
    return TypeError{
        std::string("the seoncd argument of Definectx.module must be a "
                    "source_code object.")};
  }
  const auto& custom_val = args.at(1).Get<CustomValue>();
  if (!custom_val.Has<SourceCode>()) {
    return TypeError{
        std::string("the seoncd argument of Definectx.module must be a "
                    "source_code object.")};
  }
  const auto& source_code = custom_val.Get<SourceCode>();
  return Module{func_declares, source_code};
}

Result<adt::List<ArgType>> GetFuncArgTypes(const Val& val) {
  const auto& list = CastToBuiltinValue<adt::List<Val>>(val);
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

Result<Val> DefinerCtxMakeDeclareFunc(const Val& self,
                                      const std::vector<Val>& args) {
  if (args.size() != 2) {
    return TypeError{
        std::string("Definectx.declare_func takes 2 arguments but ") +
        std::to_string(args.size()) + "were given."};
  }
  const Result<FuncId>& func_id = CastToBuiltinValue<FuncId>(args.at(0));
  ADT_RETURN_IF_ERROR(func_id);
  const Result<adt::List<ArgType>>& arg_types = GetFuncArgTypes(args.at(1));
  ADT_RETURN_IF_ERROR(arg_types);
  return FuncDeclare{func_id.GetOkValue(), arg_types.GetOkValue()};
}

Result<Val> DefinerCtxMakeSource(const Val& self,
                                 const std::vector<Val>& args) {
  if (args.size() != 1) {
    return TypeError{
        std::string("Definectx.declare_func takes 1 arguments. but ") +
        std::to_string(args.size()) + "were given."};
  }
  if (!args.at(0).Has<std::string>()) {
    return TypeError{std::string(
        "the first argument of Definectx.source_code must be string.")};
  }
  return SourceCode{args.at(0).Get<std::string>()};
}

template <typename T>
Result<Val> DefinerCtxMakeArithmeticType(const DefinerCtx<Val>& ctx,
                                         const std::string&) {
  return ArithmeticType{CppArithmeticType<T>{}};
}

template <typename T>
Result<Val> DefinerCtxMakePointerType(const DefinerCtx<Val>& ctx,
                                      const std::string&) {
  return PointerType{CppPointerType<T>{}};
}

template <BuiltinFuncType<Val> BuiltinFunc>
Result<Val> DefinerCtxMethod(const DefinerCtx<Val>& ctx, const std::string&) {
  return pexpr::Method<Val>{ctx, BuiltinFunc};
}

using DefinerCtxGetAttrT = Result<Val> (*)(const DefinerCtx<Val>& ctx,
                                           const std::string&);

Result<Val> DefinerCtxGetAttr(const DefinerCtx<Val>& ctx,
                              const std::string& name) {
  static const std::unordered_map<std::string, DefinerCtxGetAttrT> map{
      {"module", &DefinerCtxMethod<&DefinerCtxMakeModule>},
      {"declare_func", &DefinerCtxMethod<&DefinerCtxMakeDeclareFunc>},
      {"source_code", &DefinerCtxMethod<&DefinerCtxMakeSource>},
#define MAKE_CPP_TYPE_CASE(cpp_type, enum_type)                      \
  {#cpp_type, &DefinerCtxMakeArithmeticType<cpp_type>},              \
      {"const_" #cpp_type, &DefinerCtxMakeArithmeticType<cpp_type>}, \
      {#cpp_type "_ptr", &DefinerCtxMakePointerType<cpp_type*>},     \
      {"const_" #cpp_type "_ptr",                                    \
       &DefinerCtxMakePointerType<const cpp_type*>},
      PD_FOR_EACH_DATA_TYPE(MAKE_CPP_TYPE_CASE)
#undef MAKE_CPP_TYPE_CASE
#define MAKE_INT_CPP_TYPE_CASE(cpp_type)                                 \
  {#cpp_type, &DefinerCtxMakeArithmeticType<cpp_type##_t>},              \
      {"const_" #cpp_type, &DefinerCtxMakeArithmeticType<cpp_type##_t>}, \
      {#cpp_type "_ptr", &DefinerCtxMakePointerType<cpp_type##_t*>},     \
      {"const_" #cpp_type "_ptr",                                        \
       &DefinerCtxMakePointerType<const cpp_type##_t*>},
          AP_FOR_EACH_INT_TYPE(MAKE_INT_CPP_TYPE_CASE)
#undef MAKE_INT_CPP_TYPE_CASE
              {"void_ptr", &DefinerCtxMakePointerType<void*>},
      {"const_void_ptr", &DefinerCtxMakePointerType<const void*>},
  };
  const auto iter = map.find(name);
  if (iter == map.end()) {
    return AttributeError{
        std::string("'DefinerCtx' object has no attribute '") + name + "' "};
  }
  return iter->second(ctx, name);
}

Result<Val> MakeDefinerCtx(const Val& self, const std::vector<Val>& args) {
  if (args.size() != 1) {
    return adt::errors::TypeError{
        std::string() + "'DefinerRawCtx.DefinerCtx' takes 1 arguments, but " +
        std::to_string(args.size()) + "were given."};
  }

  const Result<DefinerRawCtx>& raw_ctx = CastToCustomValue<DefinerRawCtx>(self);
  ADT_RETURN_IF_ERROR(raw_ctx);
  const Result<pexpr::Object<Val>>& object = args.at(0).Match(
      [&](const pexpr::Object<Val>& obj) -> Result<pexpr::Object<Val>> {
        return obj;
      },
      [&](const pexpr::Nothing&) -> Result<pexpr::Object<Val>> {
        return pexpr::Object<Val>{};
      },
      [&](const auto&) -> Result<pexpr::Object<Val>> {
        return adt::errors::TypeError{
            std::string() +
            "the first argument of 'DefinerRawCtx.DefinerCtx' "
            "must be an object."};
      });
  ADT_RETURN_IF_ERROR(object);
  return DefinerCtx<Val>{raw_ctx.GetOkValue(), object.GetOkValue()};
}

Result<Val> DefineRawContextMakeDefinerCtx(const DefinerRawCtx& raw_ctx,
                                           const std::string&) {
  return pexpr::Method<Val>{raw_ctx, &MakeDefinerCtx};
}

using DefinerRawCtxGettAttrT = Result<Val> (*)(const DefinerRawCtx& raw_ctx,
                                               const std::string&);

Result<Val> DefinerRawCtxGetAttr(const DefinerRawCtx& raw_ctx,
                                 const std::string& name) {
  static const std::unordered_map<std::string, DefinerRawCtxGettAttrT> map{
      {"DefinerCtx", &DefineRawContextMakeDefinerCtx},
  };
  const auto& iter = map.find(name);
  if (iter == map.end()) {
    return AttributeError{std::string("'DefinerRawCtx' has no attribute '") +
                          name + "'"};
  }
  return iter->second(raw_ctx, name);
}

}  // namespace

Result<Val> CustomGetAttr(const CustomValue& custom_value,
                          const std::string& name) {
  return custom_value.Match(
      [&](const DefinerRawCtx& raw_ctx) -> Result<Val> {
        return DefinerRawCtxGetAttr(raw_ctx, name);
      },
      [&](const DefinerCtx<Val>& ctx) -> Result<Val> {
        return DefinerCtxGetAttr(ctx, name);
      },
      [&](const auto&) -> Result<Val> {
        return AttributeError{std::string("object has no attribute '") + name +
                              "' "};
      });
}

}  // namespace ap::kernel_define
