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
#include <functional>
#include "paddle/pir/include/dialect/pexpr/value.h"

namespace pexpr {

template <typename Val>
Result<Val> BuiltinIdentity(const Val&, const std::vector<Val>& args) {
  if (args.size() != 1) {
    return TypeError{std::string(kBuiltinId()) + "takes 1 argument, but " +
                     std::to_string(args.size()) + "were given."};
  }
  return args.at(0);
}

template <typename Val>
Result<Val> BuiltinList(const Val&, const std::vector<Val>& args) {
  adt::List<Val> l;
  l->reserve(args.size());
  for (const auto& arg : args) {
    l->emplace_back(arg);
  }
  return Val{l};
}

template <typename Val>
Result<Val> BuiltinGetAttr(const Val&, const std::vector<Val>& args) {
  if (args.size() != 2) {
    return TypeError{std::string(kBuiltinGetAttr()) +
                     " takes 2 argument, but " + std::to_string(args.size()) +
                     "were given."};
  }
  if (!args.at(1).template Has<std::string>()) {
    return TypeError{"attr_name must be a string"};
  }
  return ValueGetAttr(args.at(0), args.at(1).template Get<std::string>());
}

template <typename Val>
Result<Val> BuiltinGetItem(const Val&, const std::vector<Val>& args) {
  if (args.size() != 2) {
    return TypeError{std::string(kBuiltinGetItem()) +
                     " takes 2 argument, but " + std::to_string(args.size()) +
                     "were given."};
  }
  return ValueGetItem(args.at(0), args.at(1));
}

template <typename Val>
Result<Val> BuiltinIf(const InterpretFuncType<Val>& Interpret,
                      const std::vector<Val>& args) {
  if (args.size() != 3) {
    return TypeError{std::string("`if` takes 3 arguments, but ") +
                     std::to_string(args.size()) + "were given."};
  }
  const auto& cond = args.at(0);
  Result<bool> select_true_branch_res = cond.Match(
      [&](const bool c) -> Result<bool> { return c; },
      [&](const int64_t c) -> Result<bool> { return c != 0; },
      [&](const std::string& c) -> Result<bool> { return !c.empty(); },
      [&](const Nothing&) -> Result<bool> { return false; },
      [&](const adt::List<Val>& list) -> Result<bool> {
        return list->size() > 0;
      },
      [&](const Object<Val>& obj) -> Result<bool> { return obj->size() > 0; },
      [&](const Closure<Val>& closure) -> Result<bool> { return true; },
      [&](const Method<Val>& closure) -> Result<bool> { return true; },
      [&](const BuiltinFuncType<Val>& closure) -> Result<bool> { return true; },
      [&](const BuiltinHighOrderFuncType<Val>& closure) -> Result<bool> {
        return true;
      },
      [&](const auto&) -> Result<bool> {
        return TypeError{"index expr could not be a condition"};
      });
  ADT_RETURN_IF_ERROR(select_true_branch_res);
  bool select_true_branch = select_true_branch_res.GetOkValue();
  const auto& true_val = args.at(1);
  const auto& false_val = args.at(2);
  return Interpret(select_true_branch ? true_val : false_val, {});
}

template <typename Val>
Result<Val> BuiltinApply(const InterpretFuncType<Val>& Interpret,
                         const std::vector<Val>& args) {
  if (args.size() != 2) {
    return TypeError{std::string(kBuiltinApply()) + "takes 2 arguments, but " +
                     std::to_string(args.size()) + "were given."};
  }
  const auto& opt_arg_list = CastToBuiltinValue<adt::List<Val>>(args.at(1));
  ADT_RETURN_IF_ERROR(opt_arg_list);
  const auto& arg_list = opt_arg_list.GetOkValue();
  return Interpret(args.at(0), arg_list.vector());
}

template <typename Val>
Result<Val> BuiltinHalt(const Val&, const std::vector<Val>& args) {
  return RuntimeError{"Dead code. Halt function should never be touched."};
}

template <typename Val>
Result<adt::Ok> CpsBuiltinIf(CpsInterpreterBase<Val>* interpreter,
                             ComposedCallImpl<Val>* composed_call) {
  const auto args = composed_call->args;
  if (args.size() != 3) {
    return TypeError{std::string("`if` takes 3 arguments, but ") +
                     std::to_string(args.size()) + "were given."};
  }
  const auto& cond = args.at(0);
  Result<bool> select_true_branch_res = cond.Match(
      [&](const bool c) -> Result<bool> { return c; },
      [&](const int64_t c) -> Result<bool> { return c != 0; },
      [&](const std::string& c) -> Result<bool> { return !c.empty(); },
      [&](const Nothing&) -> Result<bool> { return false; },
      [&](const adt::List<Val>& list) -> Result<bool> {
        return list->size() > 0;
      },
      [&](const Object<Val>& obj) -> Result<bool> { return obj->size() > 0; },
      [&](const Closure<Val>& closure) -> Result<bool> { return true; },
      [&](const Method<Val>& closure) -> Result<bool> { return true; },
      [&](const BuiltinFuncType<Val>& closure) -> Result<bool> { return true; },
      [&](const CpsBuiltinHighOrderFuncType<Val>& closure) -> Result<bool> {
        return true;
      },
      [&](const auto&) -> Result<bool> {
        return TypeError{"index expr could not be a condition"};
      });
  ADT_RETURN_IF_ERROR(select_true_branch_res);
  bool select_true_branch = select_true_branch_res.GetOkValue();
  const auto& opt_true_closure = CastToBuiltinValue<Closure<Val>>(args.at(1));
  ADT_RETURN_IF_ERROR(opt_true_closure);
  const auto& true_closure = opt_true_closure.GetOkValue();
  const auto& opt_false_closure = CastToBuiltinValue<Closure<Val>>(args.at(2));
  ADT_RETURN_IF_ERROR(opt_false_closure);
  const auto& false_closure = opt_true_closure.GetOkValue();
  Closure<Val> closure{select_true_branch ? true_closure : false_closure};
  return interpreter->InterpretLambdaCall(closure->environment,
                                          composed_call->outter_func,
                                          closure->lambda,
                                          std::vector<Val>(),
                                          composed_call);
}

template <typename Val>
Result<adt::Ok> CpsBuiltinApply(ComposedCallImpl<Val>* composed_call) {
  const auto& args = composed_call->args;
  if (args.size() != 2) {
    return TypeError{std::string(kBuiltinApply()) + "takes 2 arguments, but " +
                     std::to_string(args.size()) + "were given."};
  }
  const auto& opt_arg_list = CastToBuiltinValue<adt::List<Val>>(args.at(1));
  if (!opt_arg_list.HasOkValue()) {
    return TypeError{std::string() + "the second arguments must be list, " +
                     GetBuiltinTypeName(args.at(1)) + " were given."};
  }
  const auto& arg_list = opt_arg_list.GetOkValue();
  composed_call->inner_func = args.at(0);
  composed_call->args = arg_list.vector();
  return adt::Ok{};
}

}  // namespace pexpr
