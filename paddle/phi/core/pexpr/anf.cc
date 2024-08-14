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

#include "paddle/phi/core/pexpr/anf.h"
#include "paddle/phi/core/pexpr/anf_builder.h"

#include <unordered_map>
#include "nlohmann/json.hpp"

namespace pexpr {

namespace {

using Json = nlohmann::json;

template <typename T>
std::string GetJsonNodeType();

template <>
std::string GetJsonNodeType<tVar<std::string>>() {
  return "symbol";
}

template <>
std::string GetJsonNodeType<bool>() {
  return "bool";
}

template <>
std::string GetJsonNodeType<int64_t>() {
  return "int64";
}

template <>
std::string GetJsonNodeType<std::string>() {
  return "string";
}

template <>
std::string GetJsonNodeType<PrimitiveOp>() {
  return "op";
}

template <>
std::string GetJsonNodeType<Lambda<AnfExpr>>() {
  return "lambda";
}

template <>
std::string GetJsonNodeType<Call<AnfExpr>>() {
  return "call";
}

template <>
std::string GetJsonNodeType<If<AnfExpr>>() {
  return "if";
}

template <>
std::string GetJsonNodeType<Let<AnfExpr>>() {
  return "let";
}

Json ConvertAnfExprToJson(const AnfExpr& anf_expr);

Json ConvertAtomicAnfExprToJson(const Atomic<AnfExpr>& atomic_expr) {
  return atomic_expr.Match(
      [&](const tVar<std::string>& var) {
        Json j;
        j["type"] = GetJsonNodeType<tVar<std::string>>();
        j["data"] = var.value();
        return j;
      },
      [&](bool c) {
        Json j;
        j["type"] = GetJsonNodeType<bool>();
        j["data"] = c;
        return j;
      },
      [&](int64_t c) {
        Json j;
        j["type"] = GetJsonNodeType<int64_t>();
        j["data"] = c;
        return j;
      },
      [&](const std::string& c) {
        Json j;
        j["type"] = GetJsonNodeType<std::string>();
        j["data"] = c;
        return j;
      },
      [&](const PrimitiveOp& c) {
        Json j;
        j["type"] = GetJsonNodeType<PrimitiveOp>();
        j["data"] = c.op_name;
        return j;
      },
      [&](const Lambda<AnfExpr>& lambda) {
        Json j;
        j["type"] = GetJsonNodeType<Lambda<AnfExpr>>();
        j["args"] = [&] {
          Json j_args = Json::array();
          for (const auto& arg : lambda.args) {
            j_args.push_back(arg.value());
          }
          return j_args;
        }();
        j["body"] = ConvertAnfExprToJson(*lambda.body);
        return j;
      });
}

Json ConvertCombinedAnfExprToJson(const Combined<AnfExpr>& combined_expr) {
  return combined_expr.Match(
      [&](const Call<AnfExpr>& call_expr) {
        Json j;
        j["type"] = GetJsonNodeType<Call<AnfExpr>>();
        j["func"] = ConvertAtomicAnfExprToJson(call_expr.func);
        j["args"] = [&] {
          Json j_args = Json::array();
          for (const auto& arg : call_expr.args) {
            j_args.push_back(ConvertAtomicAnfExprToJson(arg));
          }
          return j_args;
        }();
        return j;
      },
      [&](const If<AnfExpr>& if_expr) {
        Json j;
        j["type"] = GetJsonNodeType<If<AnfExpr>>();
        j["c"] = ConvertAtomicAnfExprToJson(if_expr.cond);
        j["t"] = ConvertAnfExprToJson(*if_expr.true_expr);
        j["f"] = ConvertAnfExprToJson(*if_expr.false_expr);
        return j;
      });
}

Json ConvertBindingAnfExprToJson(const Bind<AnfExpr>& binding_expr) {
  Json j_binding = Json::array();
  j_binding.push_back(binding_expr.var.value());
  j_binding.push_back(ConvertCombinedAnfExprToJson(binding_expr.val));
  return j_binding;
}

Json ConvertLetAnfExprToJson(const Let<AnfExpr>& let_expr) {
  Json j;
  j["type"] = GetJsonNodeType<Let<AnfExpr>>();
  j["bindings"] = [&] {
    Json j_bindings = Json::array();
    for (const auto& binding : let_expr.bindings) {
      j_bindings.push_back(ConvertBindingAnfExprToJson(binding));
    }
    return j_bindings;
  }();
  j["body"] = ConvertAnfExprToJson(*let_expr.body);
  return j;
}

Json ConvertAnfExprToJson(const AnfExpr& anf_expr) {
  return anf_expr.Match(
      [&](const Atomic<AnfExpr>& atomic_expr) {
        return ConvertAtomicAnfExprToJson(atomic_expr);
      },
      [&](const Combined<AnfExpr>& combined_expr) {
        return ConvertCombinedAnfExprToJson(combined_expr);
      },
      [&](const Let<AnfExpr>& let_expr) {
        return ConvertLetAnfExprToJson(let_expr);
      });
}

std::optional<AnfExpr> ConvertJsonToAnfExpr(const Json& j_obj);

typedef std::optional<AnfExpr> (*JsonParseFuncType)(const Json& j_obj);

template <typename T>
std::optional<AnfExpr> ParseJsonToAnfExpr(const Json& j_obj);

template <>
std::optional<AnfExpr> ParseJsonToAnfExpr<tVar<std::string>>(
    const Json& j_obj) {
  if (!j_obj.is_object()) {
    return std::nullopt;
  }
  if (!j_obj.contains("data")) {
    return std::nullopt;
  }
  if (!j_obj["data"].is_string()) {
    return std::nullopt;
  }
  std::string str = j_obj["data"].get<std::string>();
  return AnfExpr{AnfExprBuilder().Var(str)};
}

template <>
std::optional<AnfExpr> ParseJsonToAnfExpr<bool>(const Json& j_obj) {
  if (!j_obj.is_object()) {
    return std::nullopt;
  }
  if (!j_obj.contains("data")) {
    return std::nullopt;
  }
  if (!j_obj["data"].is_boolean()) {
    return std::nullopt;
  }
  bool c = j_obj["data"].get<bool>();
  return AnfExpr{AnfExprBuilder().Bool(c)};
}

template <>
std::optional<AnfExpr> ParseJsonToAnfExpr<int64_t>(const Json& j_obj) {
  if (!j_obj.is_object()) {
    return std::nullopt;
  }
  if (!j_obj.contains("data")) {
    return std::nullopt;
  }
  if (!j_obj["data"].is_number_integer()) {
    return std::nullopt;
  }
  auto c = j_obj["data"].get<Json::number_integer_t>();
  return AnfExpr{AnfExprBuilder().Int64(c)};
}

template <>
std::optional<AnfExpr> ParseJsonToAnfExpr<std::string>(const Json& j_obj) {
  if (!j_obj.is_object()) {
    return std::nullopt;
  }
  if (!j_obj.contains("data")) {
    return std::nullopt;
  }
  if (!j_obj["data"].is_string()) {
    return std::nullopt;
  }
  auto c = j_obj["data"].get<std::string>();
  return AnfExpr{AnfExprBuilder().String(c)};
}

template <>
std::optional<AnfExpr> ParseJsonToAnfExpr<PrimitiveOp>(const Json& j_obj) {
  if (!j_obj.is_object()) {
    return std::nullopt;
  }
  if (!j_obj.contains("data")) {
    return std::nullopt;
  }
  if (!j_obj["data"].is_string()) {
    return std::nullopt;
  }
  auto c = j_obj["data"].get<std::string>();
  return AnfExpr{AnfExprBuilder().PrimitiveOp(PrimitiveOp{c})};
}

template <>
std::optional<AnfExpr> ParseJsonToAnfExpr<Lambda<AnfExpr>>(const Json& j_obj) {
  if (!j_obj.is_object()) {
    return std::nullopt;
  }
  if (!j_obj.contains("args")) {
    return std::nullopt;
  }
  if (!j_obj["args"].is_array()) {
    return std::nullopt;
  }
  if (!j_obj.contains("body")) {
    return std::nullopt;
  }
  std::vector<tVar<std::string>> args;
  for (const auto& arg : j_obj["args"]) {
    if (!arg.is_string()) {
      return std::nullopt;
    }
    args.emplace_back(arg.get<std::string>());
  }
  const auto& body = ConvertJsonToAnfExpr(j_obj["body"]);
  if (!body.has_value()) {
    return std::nullopt;
  }
  return AnfExpr{AnfExprBuilder().Lambda(args, body.value())};
}

template <>
std::optional<AnfExpr> ParseJsonToAnfExpr<Call<AnfExpr>>(const Json& j_obj) {
  if (!j_obj.is_object()) {
    return std::nullopt;
  }
  if (!j_obj.contains("func")) {
    return std::nullopt;
  }
  if (!j_obj.contains("args")) {
    return std::nullopt;
  }
  if (!j_obj["args"].is_array()) {
    return std::nullopt;
  }
  const auto& func = ConvertJsonToAnfExpr(j_obj["func"]);
  if (!func.has_value()) {
    return std::nullopt;
  }
  if (!func.value().Has<Atomic<AnfExpr>>()) {
    return std::nullopt;
  }
  std::vector<Atomic<AnfExpr>> args;
  for (const auto& arg : j_obj["args"]) {
    const auto& arg_expr = ConvertJsonToAnfExpr(arg);
    if (!arg_expr.has_value()) {
      return std::nullopt;
    }
    if (!arg_expr.value().Has<Atomic<AnfExpr>>()) {
      return std::nullopt;
    }
    args.push_back(arg_expr.value().Get<Atomic<AnfExpr>>());
  }
  return AnfExpr{
      AnfExprBuilder().Call(func.value().Get<Atomic<AnfExpr>>(), args)};
}

template <>
std::optional<AnfExpr> ParseJsonToAnfExpr<If<AnfExpr>>(const Json& j_obj) {
  if (!j_obj.is_object()) {
    return std::nullopt;
  }
  if (!j_obj.contains("c")) {
    return std::nullopt;
  }
  if (!j_obj.contains("t")) {
    return std::nullopt;
  }
  if (!j_obj.contains("f")) {
    return std::nullopt;
  }
  const auto& cond = ConvertJsonToAnfExpr(j_obj["c"]);
  if (!cond.has_value()) {
    return std::nullopt;
  }
  if (!cond.value().Has<Atomic<AnfExpr>>()) {
    return std::nullopt;
  }
  const auto& cond_expr = cond.value().Get<Atomic<AnfExpr>>();
  const auto& true_expr = ConvertJsonToAnfExpr(j_obj["t"]);
  if (!true_expr.has_value()) {
    return std::nullopt;
  }
  const auto& false_expr = ConvertJsonToAnfExpr(j_obj["f"]);
  if (!false_expr.has_value()) {
    return std::nullopt;
  }
  return AnfExpr{
      AnfExprBuilder().If(cond_expr, true_expr.value(), false_expr.value())};
}

template <>
std::optional<AnfExpr> ParseJsonToAnfExpr<Let<AnfExpr>>(const Json& j_obj) {
  if (!j_obj.is_object()) {
    return std::nullopt;
  }
  if (!j_obj.contains("bindings")) {
    return std::nullopt;
  }
  if (!j_obj["bindings"].is_array()) {
    return std::nullopt;
  }
  if (!j_obj.contains("body")) {
    return std::nullopt;
  }
  std::vector<Bind<AnfExpr>> bindings;
  for (const auto& binding : j_obj["bindings"]) {
    if (!binding.is_array()) {
      return std::nullopt;
    }
    if (binding.size() != 2) {
      return std::nullopt;
    }
    if (!binding.at(0).is_string()) {
      return std::nullopt;
    }
    std::string var = binding.at(0).get<std::string>();
    const auto& val = ConvertJsonToAnfExpr(binding.at(1));
    if (!val.has_value()) {
      return std::nullopt;
    }
    if (!val.value().Has<Combined<AnfExpr>>()) {
      return std::nullopt;
    }
    bindings.push_back(
        AnfExprBuilder().Bind(var, val.value().Get<Combined<AnfExpr>>()));
  }
  const auto& body = ConvertJsonToAnfExpr(j_obj["body"]);
  if (!body.has_value()) {
    return std::nullopt;
  }
  return AnfExpr{AnfExprBuilder().Let(bindings, body.value())};
}

std::optional<JsonParseFuncType> GetJsonParseFunc(const std::string& type) {
  static const std::unordered_map<std::string, JsonParseFuncType> map = {
      {GetJsonNodeType<tVar<std::string>>(),
       &ParseJsonToAnfExpr<tVar<std::string>>},
      {GetJsonNodeType<bool>(), &ParseJsonToAnfExpr<bool>},
      {GetJsonNodeType<int64_t>(), &ParseJsonToAnfExpr<int64_t>},
      {GetJsonNodeType<std::string>(), &ParseJsonToAnfExpr<std::string>},
      {GetJsonNodeType<PrimitiveOp>(), &ParseJsonToAnfExpr<PrimitiveOp>},
      {GetJsonNodeType<Lambda<AnfExpr>>(),
       &ParseJsonToAnfExpr<Lambda<AnfExpr>>},
      {GetJsonNodeType<Call<AnfExpr>>(), &ParseJsonToAnfExpr<Call<AnfExpr>>},
      {GetJsonNodeType<If<AnfExpr>>(), &ParseJsonToAnfExpr<If<AnfExpr>>},
      {GetJsonNodeType<Let<AnfExpr>>(), &ParseJsonToAnfExpr<Let<AnfExpr>>},
  };
  const auto& iter = map.find(type);
  if (iter == map.end()) {
    return std::nullopt;
  }
  return iter->second;
}

std::optional<AnfExpr> ConvertJsonToAnfExpr(const Json& j_obj) {
  if (!j_obj.is_object()) {
    return std::nullopt;
  }
  if (!j_obj.contains("type")) {
    return std::nullopt;
  }
  const auto& parser_func = GetJsonParseFunc(j_obj["type"]);
  if (!parser_func.has_value()) {
    return std::nullopt;
  }
  return parser_func.value()(j_obj);
}

}  // namespace

std::optional<AnfExpr> AnfExpr::ParseFromJsonString(
    const std::string& json_str) {
  return ConvertJsonToAnfExpr(Json::parse(json_str));
}

std::string AnfExpr::DumpToJsonString() {
  return ConvertAnfExprToJson(*this).dump();
}

}  // namespace pexpr
