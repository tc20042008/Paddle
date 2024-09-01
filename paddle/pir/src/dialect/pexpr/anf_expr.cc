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

#include "paddle/pir/include/dialect/pexpr/anf_expr.h"
#include "paddle/pir/include/dialect/pexpr/anf_expr_builder.h"

#include <glog/logging.h>
#include <exception>
#include <unordered_map>
#include "nlohmann/json.hpp"

namespace pexpr {

namespace {

static const char kString[] = "str";
static const char kLambda[] = "lambda";
static const char kIf[] = "if";
static const char kLet[] = "__builtin_let__";

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

Result<AnfExpr> ConvertJsonToAnfExpr(const Json& j_obj);

typedef Result<AnfExpr> (*JsonParseFuncType)(const Json& j_obj);

template <typename T>
Result<AnfExpr> ParseJsonToAnfExpr(const Json& j_obj);

adt::errors::Error JsonParseFailed(const Json& j_obj, const std::string& msg) {
  return adt::errors::TypeError{msg + " json: " + j_obj.dump()};
}

adt::errors::Error JsonParseMismatch(const Json& j_obj,
                                     const std::string& msg) {
  return adt::errors::MismatchError{msg};
}

template <>
Result<AnfExpr> ParseJsonToAnfExpr<tVar<std::string>>(const Json& j_obj) {
  if (!j_obj.is_string()) {
    return JsonParseMismatch(j_obj,
                             "ParseJsonToAnfExpr<tVar<std::string>>: json "
                             "objects should be strings");
  }
  std::string str = j_obj.get<std::string>();
  return AnfExpr{AnfExprBuilder().Var(str)};
}

template <>
Result<AnfExpr> ParseJsonToAnfExpr<bool>(const Json& j_obj) {
  if (!j_obj.is_boolean()) {
    return JsonParseMismatch(
        j_obj, "ParseJsonToAnfExpr<bool>: json objects should be booleans");
  }
  bool c = j_obj.get<bool>();
  return AnfExpr{AnfExprBuilder().Bool(c)};
}

template <>
Result<AnfExpr> ParseJsonToAnfExpr<int64_t>(const Json& j_obj) {
  if (!j_obj.is_number_integer()) {
    return JsonParseMismatch(
        j_obj, "ParseJsonToAnfExpr<int64_t>: json objects should be  numbers");
  }
  auto c = j_obj.get<Json::number_integer_t>();
  return AnfExpr{AnfExprBuilder().Int64(c)};
}

template <>
Result<AnfExpr> ParseJsonToAnfExpr<std::string>(const Json& j_obj) {
  if (!j_obj.is_object()) {
    return JsonParseMismatch(j_obj,
                             "ParseJsonToAnfExpr<std::string>: an string "
                             "AnfExpr should be a json object.");
  }
  if (!j_obj.contains(kString)) {
    return JsonParseMismatch(j_obj,
                             "ParseJsonToAnfExpr<std::string>: an string "
                             "AnfExpr should contain a string.");
  }
  if (j_obj.size() != 1) {
    return JsonParseFailed(j_obj,
                           "ParseJsonToAnfExpr<std::string>: length of json "
                           "object should equal to 1.");
  }
  if (!j_obj[kString].is_string()) {
    return JsonParseFailed(j_obj,
                           "ParseJsonToAnfExpr<std::string>: an string AnfExpr "
                           "should contain a string.");
  }
  auto c = j_obj[kString].get<std::string>();
  return AnfExpr{AnfExprBuilder().String(c)};
}

template <>
Result<AnfExpr> ParseJsonToAnfExpr<Lambda<AnfExpr>>(const Json& j_obj) {
  if (!j_obj.is_array()) {
    return JsonParseMismatch(
        j_obj,
        "ParseJsonToAnfExpr<Lambda<AnfExpr>>: json objects should be arrays.");
  }
  if (j_obj.size() != 3) {
    return JsonParseMismatch(j_obj,
                             "ParseJsonToAnfExpr<Lambda<AnfExpr>>: length of "
                             "json array should equal to 3.");
  }
  if (j_obj.at(0) != kLambda) {
    return JsonParseMismatch(j_obj,
                             "ParseJsonToAnfExpr<Lambda<AnfExpr>>: the first "
                             "element of json array should equal to 'lambda'.");
  }
  if (!j_obj.at(1).is_array()) {
    return JsonParseFailed(j_obj,
                           "ParseJsonToAnfExpr<Lambda<AnfExpr>>: the second "
                           "element of json array should be a list.");
  }
  std::vector<tVar<std::string>> args;
  for (const auto& arg : j_obj.at(1)) {
    if (!arg.is_string()) {
      return JsonParseFailed(j_obj,
                             "ParseJsonToAnfExpr<Lambda<AnfExpr>>: lambda "
                             "arguments should be var names.");
    }
    args.emplace_back(arg.get<std::string>());
  }
  const auto& body = ConvertJsonToAnfExpr(j_obj.at(2));
  if (!body.HasOkValue()) {
    return JsonParseFailed(j_obj,
                           "ParseJsonToAnfExpr<Lambda<AnfExpr>>: the lambda "
                           "body should be a valid AnfExpr.");
  }
  return AnfExpr{AnfExprBuilder().Lambda(args, body.GetOkValue())};
}

template <>
Result<AnfExpr> ParseJsonToAnfExpr<Call<AnfExpr>>(const Json& j_obj) {
  if (!j_obj.is_array()) {
    return JsonParseMismatch(
        j_obj,
        "ParseJsonToAnfExpr<Call<AnfExpr>>: json objects should be arrays.");
  }
  if (j_obj.empty()) {
    return JsonParseFailed(
        j_obj,
        "ParseJsonToAnfExpr<Call<AnfExpr>>: json arrays should be not empty.");
  }
  const auto& func = ConvertJsonToAnfExpr(j_obj.at(0));
  if (!func.HasOkValue()) {
    return JsonParseFailed(j_obj,
                           "ParseJsonToAnfExpr<Call<AnfExpr>>: the function "
                           "should a valid AnfExpr.");
  }
  if (!func.GetOkValue().Has<Atomic<AnfExpr>>()) {
    return JsonParseFailed(j_obj,
                           "ParseJsonToAnfExpr<Call<AnfExpr>>: the function "
                           "should a valid atomic AnfExpr.");
  }
  std::vector<Atomic<AnfExpr>> args;
  for (int i = 1; i < j_obj.size(); ++i) {
    const auto& arg = j_obj.at(i);
    const auto& arg_expr = ConvertJsonToAnfExpr(arg);
    if (!arg_expr.HasOkValue()) {
      return JsonParseFailed(j_obj,
                             "ParseJsonToAnfExpr<Call<AnfExpr>>: the args "
                             "should be valid AnfExprs.");
    }
    if (!arg_expr.GetOkValue().Has<Atomic<AnfExpr>>()) {
      return JsonParseFailed(j_obj,
                             "ParseJsonToAnfExpr<Call<AnfExpr>>: the args "
                             "should be valid atomic AnfExprs.");
    }
    args.push_back(arg_expr.GetOkValue().Get<Atomic<AnfExpr>>());
  }
  return AnfExpr{
      AnfExprBuilder().Call(func.GetOkValue().Get<Atomic<AnfExpr>>(), args)};
}

template <>
Result<AnfExpr> ParseJsonToAnfExpr<If<AnfExpr>>(const Json& j_obj) {
  if (!j_obj.is_array()) {
    return JsonParseMismatch(j_obj,
                             "ParseJsonToAnfExpr<If<AnfExpr>>: json objects "
                             "should be valid atomic AnfExprs.");
  }
  if (j_obj.size() != 4) {
    return JsonParseMismatch(j_obj,
                             "ParseJsonToAnfExpr<If<AnfExpr>>: the length of "
                             "json array should equal to 4.");
  }
  if (j_obj.at(0) != kIf) {
    return JsonParseMismatch(j_obj,
                             "ParseJsonToAnfExpr<If<AnfExpr>>: the first "
                             "argument of json array should equal to 'if'.");
  }
  const auto& cond = ConvertJsonToAnfExpr(j_obj.at(1));
  if (!cond.HasOkValue()) {
    return JsonParseFailed(j_obj,
                           "ParseJsonToAnfExpr<If<AnfExpr>>: the second "
                           "argument of json array should a valid AnfExpr.");
  }
  if (!cond.GetOkValue().Has<Atomic<AnfExpr>>()) {
    return JsonParseFailed(
        j_obj,
        "ParseJsonToAnfExpr<If<AnfExpr>>: the second argument of json array "
        "should a valid atomic AnfExpr.");
  }
  const auto& cond_expr = cond.GetOkValue().Get<Atomic<AnfExpr>>();
  const auto& true_expr = ConvertJsonToAnfExpr(j_obj.at(2));
  if (!true_expr.HasOkValue()) {
    return JsonParseFailed(j_obj,
                           "ParseJsonToAnfExpr<If<AnfExpr>>: the third "
                           "argument of json array should a valid AnfExpr.");
  }
  const auto& false_expr = ConvertJsonToAnfExpr(j_obj.at(3));
  if (!false_expr.HasOkValue()) {
    return JsonParseFailed(j_obj,
                           "ParseJsonToAnfExpr<If<AnfExpr>>: the forth "
                           "argument of json array should a valid AnfExpr.");
  }
  return AnfExpr{AnfExprBuilder().If(
      cond_expr, true_expr.GetOkValue(), false_expr.GetOkValue())};
}

template <>
Result<AnfExpr> ParseJsonToAnfExpr<Let<AnfExpr>>(const Json& j_obj) {
  if (!j_obj.is_array()) {
    return JsonParseMismatch(
        j_obj,
        "ParseJsonToAnfExpr<Let<AnfExpr>>: json objects should be arrays.");
  }
  if (j_obj.size() != 3) {
    return JsonParseMismatch(j_obj,
                             "ParseJsonToAnfExpr<Let<AnfExpr>>: the length of "
                             "json array should equal to 3.");
  }
  if (j_obj.at(0) != kLet) {
    return JsonParseMismatch(j_obj,
                             "ParseJsonToAnfExpr<Let<AnfExpr>>: the first "
                             "argument of json array should be 'let'.");
  }
  std::vector<Bind<AnfExpr>> bindings;
  const auto& j_bindings = j_obj.at(1);
  for (int i = 0; i < j_bindings.size(); ++i) {
    const auto& binding = j_bindings.at(i);
    if (!binding.is_array()) {
      return JsonParseFailed(
          binding,
          "ParseJsonToAnfExpr<Let<AnfExpr>>: bindings should be json arrays.");
    }
    if (binding.size() != 2) {
      return JsonParseFailed(binding,
                             "ParseJsonToAnfExpr<Let<AnfExpr>>: the size of "
                             "one binding should equal to 2.");
    }
    if (!binding.at(0).is_string()) {
      return JsonParseFailed(binding.at(0),
                             "ParseJsonToAnfExpr<Let<AnfExpr>>: the first "
                             "element of a binding should be var name.");
    }
    std::string var = binding.at(0).get<std::string>();
    const auto& val = ConvertJsonToAnfExpr(binding.at(1));
    if (!val.HasOkValue()) {
      return JsonParseFailed(binding.at(1),
                             "ParseJsonToAnfExpr<Let<AnfExpr>>: the second "
                             "element of a binding should be a valid AnfExpr.");
    }
    if (!val.GetOkValue().Has<Combined<AnfExpr>>()) {
      return JsonParseFailed(
          binding.at(1),
          "ParseJsonToAnfExpr<Let<AnfExpr>>: the second element of a binding "
          "should be a valid combined AnfExpr.");
    }
    bindings.push_back(
        AnfExprBuilder().Bind(var, val.GetOkValue().Get<Combined<AnfExpr>>()));
  }
  const auto& body = ConvertJsonToAnfExpr(j_obj.at(2));
  if (!body.HasOkValue()) {
    return JsonParseFailed(j_obj.at(2),
                           "ParseJsonToAnfExpr<Let<AnfExpr>>: the body of Let "
                           "AnfExpr should be a valid AnfExpr.");
  }
  return AnfExpr{AnfExprBuilder().Let(bindings, body.GetOkValue())};
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
  };
  return vec;
}

Result<AnfExpr> ConvertJsonToAnfExpr(const Json& j_obj) {
  try {
    for (const auto& parse_func : GetJsonParseFuncs()) {
      const auto& ret = parse_func(j_obj);
      if (ret.HasOkValue()) {
        return ret.GetOkValue();
      }
      if (!ret.GetError().Has<adt::errors::MismatchError>()) {
        LOG(ERROR) << "ConvertJsonToAnfExpr: error-type: "
                   << ret.GetError().class_name()
                   << ", error-msg: " << ret.GetError().msg();
        return ret.GetError();
      }
    }
  } catch (std::exception& e) {
    return JsonParseFailed(j_obj,
                           "ConvertJsonToAnfExpr: throw error when parsing.");
  }
  return JsonParseFailed(j_obj, "ConvertJsonToAnfExpr: failed to convert.");
}

}  // namespace

Result<AnfExpr> AnfExpr::ParseFromJsonString(const std::string& json_str) {
  return ConvertJsonToAnfExpr(Json::parse(json_str));
}

std::string AnfExpr::DumpToJsonString() {
  return ConvertAnfExprToJson(*this).dump();
}

std::string AnfExpr::DumpToJsonString(int indent) {
  return ConvertAnfExprToJson(*this).dump(indent);
}

}  // namespace pexpr
