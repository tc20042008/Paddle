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

using pexpr::BuiltinFuncType;
using pexpr::CastToBuiltinValue;

adt::Result<ArgType> ArgType::MakeFromPhiDataType(::phi::DataType data_type) {
  static const std::unordered_map<::phi::DataType, ArgType> map{
#define MAKE_PHI_DATA_TYPE_TO_ARG_TYPE_CASE(cpp_type, enum_type) \
  {phi::enum_type, ArgType{CppArgType<cpp_type>{}}},
      PD_FOR_EACH_DATA_TYPE(MAKE_PHI_DATA_TYPE_TO_ARG_TYPE_CASE)
#undef MAKE_PHI_DATA_TYPE_TO_ARG_TYPE_CASE
  };
  const auto& iter = map.find(data_type);
  if (iter == map.end()) {
    return InvalidArgumentError{"Invalid phi data type."};
  }
  return iter->second;
}

namespace {

template <typename T>
struct TypeConverter;

#define SPECIALIZE_TYPE_CONVERTER(cpp_type, enum_type) \
  template <>                                          \
  struct TypeConverter<CppArgType<cpp_type>> {         \
    using remove_const_type = CppArgType<cpp_type>;    \
  };                                                   \
  template <>                                          \
  struct TypeConverter<CppArgType<cpp_type*>> {        \
    using remove_const_type = CppArgType<cpp_type*>;   \
  };                                                   \
  template <>                                          \
  struct TypeConverter<CppArgType<const cpp_type*>> {  \
    using remove_const_type = CppArgType<cpp_type*>;   \
  };

PD_FOR_EACH_DATA_TYPE(SPECIALIZE_TYPE_CONVERTER);
#undef SPECIALIZE_TYPE_CONVERTER

template <>
struct TypeConverter<CppArgType<void*>> {
  using remove_const_type = CppArgType<void*>;
};

template <>
struct TypeConverter<CppArgType<const void*>> {
  using remove_const_type = CppArgType<void*>;
};

}  // namespace

ArgType ArgType::RemoveConst() const {
  return Match([](auto impl) {
    return ArgType{typename TypeConverter<decltype(impl)>::remove_const_type{}};
  });
}

namespace {

Result<Val> MakeDefinerCtxModule(const std::vector<Val>& args) {
  if (args.size() != 3) {
    return TypeError{
        std::string(
            "Definectx.module takes 3 arguments (including self) but ") +
        std::to_string(args.size()) + "were given."};
  }
  const auto& list = args.at(1).Match(
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
  if (!args.at(2).Has<CustomValue>()) {
    return TypeError{
        std::string("the seoncd argument of Definectx.module must be a "
                    "source_code object.")};
  }
  const auto& custom_val = args.at(2).Get<CustomValue>();
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
    const auto& arg_type = CastToCustomValue<ArgType>(elt);
    ADT_RETURN_IF_ERROR(arg_type);
    ret->emplace_back(arg_type.GetOkValue());
  }
  return ret;
}

Result<Val> MakeDefinerCtxDeclareFunc(const std::vector<Val>& args) {
  if (args.size() != 3) {
    return TypeError{
        std::string(
            "Definectx.declare_func takes 3 arguments  (including self) but ") +
        std::to_string(args.size()) + "were given."};
  }
  const Result<FuncId>& func_id = CastToBuiltinValue<FuncId>(args.at(1));
  ADT_RETURN_IF_ERROR(func_id);
  const Result<adt::List<ArgType>>& arg_types = GetFuncArgTypes(args.at(2));
  ADT_RETURN_IF_ERROR(arg_types);
  return FuncDeclare{func_id.GetOkValue(), arg_types.GetOkValue()};
}

Result<Val> MakeDefinerCtxSource(const std::vector<Val>& args) {
  if (args.size() != 2) {
    return TypeError{
        std::string(
            "Definectx.declare_func takes 2 arguments  (including self) but ") +
        std::to_string(args.size()) + "were given."};
  }
  if (!args.at(1).Has<std::string>()) {
    return TypeError{std::string(
        "the first argument of Definectx.source_code must be string.")};
  }
  return SourceCode{args.at(1).Get<std::string>()};
}

template <typename T>
Result<Val> MakeDefinerCtxArgType(const DefinerCtx<Val>& ctx,
                                  const std::string&) {
  return ArgType{CppArgType<T>{}};
}

template <BuiltinFuncType<Val> BuiltinFunc>
Result<Val> DefinerCtxMethod(const DefinerCtx<Val>& ctx, const std::string&) {
  return pexpr::Method<Val>{ctx, BuiltinFuncType<Val>{BuiltinFunc}};
}

using DefinerCtxGetAttrT = Result<Val> (*)(const DefinerCtx<Val>& ctx,
                                           const std::string&);

Result<Val> DefinerCtxGetAttr(const DefinerCtx<Val>& ctx,
                              const std::string& name) {
  static const std::unordered_map<std::string, DefinerCtxGetAttrT> map{
      {"module", &DefinerCtxMethod<&MakeDefinerCtxModule>},
      {"declare_func", &DefinerCtxMethod<&MakeDefinerCtxDeclareFunc>},
      {"source_code", &DefinerCtxMethod<&MakeDefinerCtxSource>},
#define MAKE_CPP_TYPE_CASE(cpp_type, enum_type)               \
  {#cpp_type, &MakeDefinerCtxArgType<cpp_type>},              \
      {"const_" #cpp_type, &MakeDefinerCtxArgType<cpp_type>}, \
      {#cpp_type "_ptr", &MakeDefinerCtxArgType<cpp_type*>},  \
      {"const_" #cpp_type "_ptr", &MakeDefinerCtxArgType<const cpp_type*>},
      PD_FOR_EACH_DATA_TYPE(MAKE_CPP_TYPE_CASE)
#undef MAKE_CPP_TYPE_CASE
#define MAKE_INT_CPP_TYPE_CASE(cpp_type)                          \
  {#cpp_type, &MakeDefinerCtxArgType<cpp_type##_t>},              \
      {"const_" #cpp_type, &MakeDefinerCtxArgType<cpp_type##_t>}, \
      {#cpp_type "_ptr", &MakeDefinerCtxArgType<cpp_type##_t*>},  \
      {"const_" #cpp_type "_ptr",                                 \
       &MakeDefinerCtxArgType<const cpp_type##_t*>},
          AP_FOR_EACH_INT_TYPE(MAKE_INT_CPP_TYPE_CASE)
#undef MAKE_INT_CPP_TYPE_CASE
              {"void_ptr", &MakeDefinerCtxArgType<void*>},
      {"const_void_ptr", &MakeDefinerCtxArgType<const void*>},
  };
  const auto iter = map.find(name);
  if (iter == map.end()) {
    return AttributeError{
        std::string("'DefinerCtx' object has no attribute '") + name + "' "};
  }
  return iter->second(ctx, name);
}

Result<Val> MakeDefinerCtx(const std::vector<Val>& args) {
  if (args.size() != 2) {
    return adt::errors::TypeError{
        std::string() + "'DefinerRawCtx.DefinerCtx' takes 2 arguments, but " +
        std::to_string(args.size()) + "were given."};
  }

  const Result<DefinerRawCtx>& raw_ctx =
      CastToCustomValue<DefinerRawCtx>(args.at(0));
  ADT_RETURN_IF_ERROR(raw_ctx);
  const Result<pexpr::Object<Val>>& object = args.at(1).Match(
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
      [&](const ArgType&) -> Result<Val> {
        return AttributeError{
            std::string("'ArgType' object has no attribute '") + name + "' "};
      },
      [&](const auto&) -> Result<Val> {
        return AttributeError{std::string("object has no attribute '") + name +
                              "' "};
      });
}

}  // namespace ap::kernel_define
