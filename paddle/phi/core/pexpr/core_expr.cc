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

#include "paddle/phi/core/pexpr/core_expr.h"
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include "nlohmann/json.hpp"
#include "paddle/phi/core/pexpr/core_expr_builder.h"

namespace pexpr {

const char kBuiltinId[] = "__builtin_identity__";

namespace {

std::string AtomicExprToSExpression(const Atomic<CoreExpr>& core_expr) {
  return core_expr.Match(
      [](const tVar<std::string>& var) { return var.value(); },
      [](const bool c) { return c ? std::string("#t") : std::string("#f"); },
      [](const int64_t c) { return std::to_string(c); },
      [](const std::string& str) {
        std::ostringstream ss;
        ss << std::quoted(str);
        return ss.str();
      },
      [](const PrimitiveOp& op) {
        return std::string("(op ") + op.op_name + ")";
      },
      [](const Lambda<CoreExpr>& lambda) {
        std::ostringstream ss;
        ss << "(lambda [";
        int i = 0;
        for (const auto& arg : lambda.args) {
          if (i++ > 0) {
            ss << " ";
          }
          ss << arg.value();
        }
        ss << "] ";
        ss << lambda.body->ToSExpression();
        ss << ")";
        return ss.str();
      });
}

std::string ComposedCallExprToSExpression(
    const ComposedCall<CoreExpr>& core_expr) {
  std::ostringstream ss;
  ss << "(";
  ss << AtomicExprToSExpression(core_expr.outter_func);
  ss << " ";
  ss << AtomicExprToSExpression(core_expr.inner_func);
  for (const auto& arg : core_expr.args) {
    ss << " ";
    ss << AtomicExprToSExpression(arg);
  }
  ss << ")";
  return ss.str();
}

}  // namespace

std::string CoreExpr::ToSExpression() const {
  return Match(
      [&](const Atomic<CoreExpr>& core_expr) {
        return AtomicExprToSExpression(core_expr);
      },
      [&](const ComposedCall<CoreExpr>& core_expr) {
        return ComposedCallExprToSExpression(core_expr);
      });
}

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
std::string GetJsonNodeType<Lambda<CoreExpr>>() {
  return "lambda";
}

template <>
std::string GetJsonNodeType<ComposedCall<CoreExpr>>() {
  return "composed_call";
}

Json ConvertCoreExprToJson(const CoreExpr& core_expr);

Json ConvertAtomicCoreExprToJson(const Atomic<CoreExpr>& atomic_expr) {
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
      [&](const Lambda<CoreExpr>& lambda) {
        Json j;
        j["type"] = GetJsonNodeType<Lambda<CoreExpr>>();
        j["args"] = [&] {
          Json j_args = Json::array();
          for (const auto& arg : lambda.args) {
            j_args.push_back(arg.value());
          }
          return j_args;
        }();
        j["body"] = ConvertCoreExprToJson(*lambda.body);
        return j;
      });
}

Json ConvertCoreExprToJson(const CoreExpr& core_expr) {
  return core_expr.Match(
      [&](const Atomic<CoreExpr>& atomic_expr) {
        return ConvertAtomicCoreExprToJson(atomic_expr);
      },
      [&](const ComposedCall<CoreExpr>& call_expr) {
        Json j;
        j["type"] = GetJsonNodeType<ComposedCall<CoreExpr>>();
        j["outter_func"] = ConvertAtomicCoreExprToJson(call_expr.outter_func);
        j["inner_func"] = ConvertAtomicCoreExprToJson(call_expr.inner_func);
        j["args"] = [&] {
          Json j_args = Json::array();
          for (const auto& arg : call_expr.args) {
            j_args.push_back(ConvertAtomicCoreExprToJson(arg));
          }
          return j_args;
        }();
        return j;
      });
}

std::optional<CoreExpr> ConvertJsonToCoreExpr(const Json& j_obj);

typedef std::optional<CoreExpr> (*JsonParseFuncType)(const Json& j_obj);

template <typename T>
std::optional<CoreExpr> ParseJsonToCoreExpr(const Json& j_obj);

template <>
std::optional<CoreExpr> ParseJsonToCoreExpr<tVar<std::string>>(
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
  return CoreExpr{CoreExprBuilder().Var(str)};
}

template <>
std::optional<CoreExpr> ParseJsonToCoreExpr<bool>(const Json& j_obj) {
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
  return CoreExpr{CoreExprBuilder().Bool(c)};
}

template <>
std::optional<CoreExpr> ParseJsonToCoreExpr<int64_t>(const Json& j_obj) {
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
  return CoreExpr{CoreExprBuilder().Int64(c)};
}

template <>
std::optional<CoreExpr> ParseJsonToCoreExpr<std::string>(const Json& j_obj) {
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
  return CoreExpr{CoreExprBuilder().String(c)};
}

template <>
std::optional<CoreExpr> ParseJsonToCoreExpr<PrimitiveOp>(const Json& j_obj) {
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
  return CoreExpr{CoreExprBuilder().PrimitiveOp(PrimitiveOp{c})};
}

template <>
std::optional<CoreExpr> ParseJsonToCoreExpr<Lambda<CoreExpr>>(
    const Json& j_obj) {
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
  const auto& body = ConvertJsonToCoreExpr(j_obj["body"]);
  if (!body.has_value()) {
    return std::nullopt;
  }
  return CoreExpr{CoreExprBuilder().Lambda(args, body.value())};
}

template <>
std::optional<CoreExpr> ParseJsonToCoreExpr<ComposedCall<CoreExpr>>(
    const Json& j_obj) {
  if (!j_obj.is_object()) {
    return std::nullopt;
  }
  if (!j_obj.contains("outter_func")) {
    return std::nullopt;
  }
  if (!j_obj.contains("inner_func")) {
    return std::nullopt;
  }
  if (!j_obj.contains("args")) {
    return std::nullopt;
  }
  if (!j_obj["args"].is_array()) {
    return std::nullopt;
  }
  const auto& outter_func = ConvertJsonToCoreExpr(j_obj["outter_func"]);
  if (!outter_func.has_value()) {
    return std::nullopt;
  }
  if (!outter_func.value().Has<Atomic<CoreExpr>>()) {
    return std::nullopt;
  }
  const auto& inner_func = ConvertJsonToCoreExpr(j_obj["inner_func"]);
  if (!inner_func.has_value()) {
    return std::nullopt;
  }
  if (!inner_func.value().Has<Atomic<CoreExpr>>()) {
    return std::nullopt;
  }
  std::vector<Atomic<CoreExpr>> args;
  for (const auto& arg : j_obj["args"]) {
    const auto& arg_expr = ConvertJsonToCoreExpr(arg);
    if (!arg_expr.has_value()) {
      return std::nullopt;
    }
    if (!arg_expr.value().Has<Atomic<CoreExpr>>()) {
      return std::nullopt;
    }
    args.push_back(arg_expr.value().Get<Atomic<CoreExpr>>());
  }
  const auto& outter_fn = outter_func.value().Get<Atomic<CoreExpr>>();
  const auto& inner_fn = inner_func.value().Get<Atomic<CoreExpr>>();
  return CoreExpr{CoreExprBuilder().ComposedCall(outter_fn, inner_fn, args)};
}

std::optional<JsonParseFuncType> GetJsonParseFunc(const std::string& type) {
  static const std::unordered_map<std::string, JsonParseFuncType> map = {
      {GetJsonNodeType<tVar<std::string>>(),
       &ParseJsonToCoreExpr<tVar<std::string>>},
      {GetJsonNodeType<bool>(), &ParseJsonToCoreExpr<bool>},
      {GetJsonNodeType<int64_t>(), &ParseJsonToCoreExpr<int64_t>},
      {GetJsonNodeType<std::string>(), &ParseJsonToCoreExpr<std::string>},
      {GetJsonNodeType<PrimitiveOp>(), &ParseJsonToCoreExpr<PrimitiveOp>},
      {GetJsonNodeType<Lambda<CoreExpr>>(),
       &ParseJsonToCoreExpr<Lambda<CoreExpr>>},
      {GetJsonNodeType<ComposedCall<CoreExpr>>(),
       &ParseJsonToCoreExpr<ComposedCall<CoreExpr>>},
  };
  const auto& iter = map.find(type);
  if (iter == map.end()) {
    return std::nullopt;
  }
  return iter->second;
}

std::optional<CoreExpr> ConvertJsonToCoreExpr(const Json& j_obj) {
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

std::optional<CoreExpr> CoreExpr::ParseFromJsonString(
    const std::string& json_str) {
  return ConvertJsonToCoreExpr(Json::parse(json_str));
}

std::string CoreExpr::DumpToJsonString() {
  return ConvertCoreExprToJson(*this).dump();
}

}  // namespace pexpr
