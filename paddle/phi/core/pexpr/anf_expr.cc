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

#include "paddle/phi/core/pexpr/anf_expr.h"
#include "paddle/phi/core/pexpr/anf_expr_builder.h"

#include <unordered_map>
#include "nlohmann/json.hpp"

namespace pexpr {

namespace {

static const char kString[] = "str";
static const char kPrimitiveOp[] = "op";
static const char kLambda[] = "lambda";
static const char kIf[] = "if";
static const char kLet[] = "let";

using Json = nlohmann::json;

Json ConvertAnfExprToJson(const AnfExpr& anf_expr);

Json ConvertAtomicAnfExprToJson(const Atomic<AnfExpr>& atomic_expr) {
  return atomic_expr.Match(
      [&](const tVar<std::string>& var) {
        Json j = var.value();
        return j;
      },
      [&](bool c) {
        Json j = c;
        return j;
      },
      [&](int64_t c) {
        Json j = c;
        return j;
      },
      [&](const std::string& c) {
        Json j;
        j[kString] = c;
        return j;
      },
      [&](const PrimitiveOp& c) {
        Json j;
        j[kPrimitiveOp] = c.op_name;
        return j;
      },
      [&](const Lambda<AnfExpr>& lambda) {
        Json j = Json::array();
        j.push_back(kLambda);
        j.push_back([&] {
          Json j_args = Json::array();
          for (const auto& arg : lambda->args) {
            j_args.push_back(arg.value());
          }
          return j_args;
        }());
        j.push_back(ConvertAnfExprToJson(lambda->body));
        return j;
      });
}

Json ConvertCombinedAnfExprToJson(const Combined<AnfExpr>& combined_expr) {
  return combined_expr.Match(
      [&](const Call<AnfExpr>& call_expr) {
        Json j;
        j.push_back(ConvertAtomicAnfExprToJson(call_expr->func));
        for (const auto& arg : call_expr->args) {
          j.push_back(ConvertAtomicAnfExprToJson(arg));
        }
        return j;
      },
      [&](const If<AnfExpr>& if_expr) {
        Json j;
        j.push_back(kIf);
        j.push_back(ConvertAtomicAnfExprToJson(if_expr->cond));
        j.push_back(ConvertAnfExprToJson(if_expr->true_expr));
        j.push_back(ConvertAnfExprToJson(if_expr->false_expr));
        return j;
      });
}

Json ConvertBindingAnfExprToJson(const Bind<AnfExpr>& binding_expr) {
  Json j = Json::array();
  j.push_back(binding_expr.var.value());
  j.push_back(ConvertCombinedAnfExprToJson(binding_expr.val));
  return j;
}

Json ConvertLetAnfExprToJson(const Let<AnfExpr>& let_expr) {
  Json j;
  j.push_back(kLet);
  for (const auto& binding : let_expr->bindings) {
    j.push_back(ConvertBindingAnfExprToJson(binding));
  }
  j.push_back(ConvertAnfExprToJson(let_expr->body));
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
  if (!j_obj.is_string()) {
    return std::nullopt;
  }
  std::string str = j_obj.get<std::string>();
  return AnfExpr{AnfExprBuilder().Var(str)};
}

template <>
std::optional<AnfExpr> ParseJsonToAnfExpr<bool>(const Json& j_obj) {
  if (!j_obj.is_boolean()) {
    return std::nullopt;
  }
  bool c = j_obj.get<bool>();
  return AnfExpr{AnfExprBuilder().Bool(c)};
}

template <>
std::optional<AnfExpr> ParseJsonToAnfExpr<int64_t>(const Json& j_obj) {
  if (!j_obj.is_number_integer()) {
    return std::nullopt;
  }
  auto c = j_obj.get<Json::number_integer_t>();
  return AnfExpr{AnfExprBuilder().Int64(c)};
}

template <>
std::optional<AnfExpr> ParseJsonToAnfExpr<std::string>(const Json& j_obj) {
  if (!j_obj.is_object()) {
    return std::nullopt;
  }
  if (j_obj.size() != 1) {
    return std::nullopt;
  }
  if (!j_obj.contains(kString)) {
    return std::nullopt;
  }
  if (!j_obj[kString].is_string()) {
    return std::nullopt;
  }
  auto c = j_obj[kString].get<std::string>();
  return AnfExpr{AnfExprBuilder().String(c)};
}

template <>
std::optional<AnfExpr> ParseJsonToAnfExpr<PrimitiveOp>(const Json& j_obj) {
  if (!j_obj.is_object()) {
    return std::nullopt;
  }
  if (j_obj.size() != 1) {
    return std::nullopt;
  }
  if (!j_obj.contains(kPrimitiveOp)) {
    return std::nullopt;
  }
  if (!j_obj[kPrimitiveOp].is_string()) {
    return std::nullopt;
  }
  auto c = j_obj[kPrimitiveOp].get<std::string>();
  return AnfExpr{AnfExprBuilder().PrimitiveOp(PrimitiveOp{c})};
}

template <>
std::optional<AnfExpr> ParseJsonToAnfExpr<Lambda<AnfExpr>>(const Json& j_obj) {
  if (!j_obj.is_array()) {
    return std::nullopt;
  }
  if (j_obj.size() != 3) {
    return std::nullopt;
  }
  if (j_obj.at(0) != kLambda) {
    return std::nullopt;
  }
  if (!j_obj.at(1).is_array()) {
    return std::nullopt;
  }
  std::vector<tVar<std::string>> args;
  for (const auto& arg : j_obj.at(1)) {
    if (!arg.is_string()) {
      return std::nullopt;
    }
    args.emplace_back(arg.get<std::string>());
  }
  const auto& body = ConvertJsonToAnfExpr(j_obj.at(2));
  if (!body.has_value()) {
    return std::nullopt;
  }
  return AnfExpr{AnfExprBuilder().Lambda(args, body.value())};
}

template <>
std::optional<AnfExpr> ParseJsonToAnfExpr<Call<AnfExpr>>(const Json& j_obj) {
  if (!j_obj.is_array()) {
    return std::nullopt;
  }
  if (j_obj.empty()) {
    return std::nullopt;
  }
  const auto& func = ConvertJsonToAnfExpr(j_obj.at(0));
  if (!func.has_value()) {
    return std::nullopt;
  }
  if (!func.value().Has<Atomic<AnfExpr>>()) {
    return std::nullopt;
  }
  std::vector<Atomic<AnfExpr>> args;
  for (int i = 1; i < j_obj.size(); ++i) {
    const auto& arg = j_obj.at(i);
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
  if (!j_obj.is_array()) {
    return std::nullopt;
  }
  if (j_obj.size() != 4) {
    return std::nullopt;
  }
  if (j_obj.at(0) != kIf) {
    return std::nullopt;
  }
  const auto& cond = ConvertJsonToAnfExpr(j_obj.at(1));
  if (!cond.has_value()) {
    return std::nullopt;
  }
  if (!cond.value().Has<Atomic<AnfExpr>>()) {
    return std::nullopt;
  }
  const auto& cond_expr = cond.value().Get<Atomic<AnfExpr>>();
  const auto& true_expr = ConvertJsonToAnfExpr(j_obj.at(2));
  if (!true_expr.has_value()) {
    return std::nullopt;
  }
  const auto& false_expr = ConvertJsonToAnfExpr(j_obj.at(3));
  if (!false_expr.has_value()) {
    return std::nullopt;
  }
  return AnfExpr{
      AnfExprBuilder().If(cond_expr, true_expr.value(), false_expr.value())};
}

template <>
std::optional<AnfExpr> ParseJsonToAnfExpr<Let<AnfExpr>>(const Json& j_obj) {
  if (!j_obj.is_array()) {
    return std::nullopt;
  }
  if (j_obj.size() < 2) {
    return std::nullopt;
  }
  if (j_obj.at(0) != kLet) {
    return std::nullopt;
  }
  std::vector<Bind<AnfExpr>> bindings;
  for (int i = 1; i < j_obj.size() - 1; ++i) {
    const auto& binding = j_obj.at(i);
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
  const auto& body = ConvertJsonToAnfExpr(j_obj.at(j_obj.size() - 1));
  if (!body.has_value()) {
    return std::nullopt;
  }
  return AnfExpr{AnfExprBuilder().Let(bindings, body.value())};
}

const std::vector<JsonParseFuncType>& GetJsonParseFuncs() {
  static const std::vector<JsonParseFuncType> vec{
      &ParseJsonToAnfExpr<Lambda<AnfExpr>>,
      &ParseJsonToAnfExpr<If<AnfExpr>>,
      &ParseJsonToAnfExpr<Let<AnfExpr>>,
      &ParseJsonToAnfExpr<Call<AnfExpr>>,
      &ParseJsonToAnfExpr<tVar<std::string>>,
      &ParseJsonToAnfExpr<bool>,
      &ParseJsonToAnfExpr<int64_t>,
      &ParseJsonToAnfExpr<std::string>,
      &ParseJsonToAnfExpr<PrimitiveOp>,
  };
  return vec;
}

std::optional<AnfExpr> ConvertJsonToAnfExpr(const Json& j_obj) {
  for (const auto& parse_func : GetJsonParseFuncs()) {
    if (const auto& ret = parse_func(j_obj)) {
      return ret;
    }
  }
  return std::nullopt;
}

}  // namespace

std::optional<AnfExpr> AnfExpr::ParseFromJsonString(
    const std::string& json_str) {
  return ConvertJsonToAnfExpr(Json::parse(json_str));
}

std::string AnfExpr::DumpToJsonString() {
  return ConvertAnfExprToJson(*this).dump();
}

std::string AnfExpr::DumpToJsonString(int indent) {
  return ConvertAnfExprToJson(*this).dump(indent);
}

}  // namespace pexpr
