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
Result<Val> BuiltinIf(const InterpretFuncType<Val>& Interpret,
                      const std::vector<Val>& args) {
  if (args.size() != 3) {
    return InvalidArgumentError{std::string("`if` takes 3 arguments, but ") +
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
      [&](const BuiltinFuncType<Val>& closure) -> Result<bool> { return true; },
      [&](const auto&) -> Result<bool> {
        return InvalidArgumentError{"index expr could not be a condition"};
      });
  if (select_true_branch_res.Has<Error>()) {
    return select_true_branch_res.Get<Error>();
  }
  bool select_true_branch = select_true_branch_res.Get<bool>();
  const auto& true_val = args.at(1);
  const auto& false_val = args.at(2);
  return std::visit(
      ::common::Overloaded{
          [&](const Closure<Val>& true_closure,
              const Closure<Val>& false_val) -> Result<Val> {
            return Interpret(select_true_branch ? true_closure : false_val, {});
          },
          [&](const auto&, const auto&) -> Result<Val> {
            return TypeError{"true_branch or false_branch is not closure"};
          }},
      true_val.variant(),
      false_val.variant());
}

template <typename Val>
Result<Val> BuiltinIdentity(const InterpretFuncType<Val>& Interpret,
                            const std::vector<Val>& args) {
  if (args.size() != 1) {
    return InvalidArgumentError{std::string(kBuiltinId) +
                                "takes 1 argument, but " +
                                std::to_string(args.size()) + "were given."};
  }
  return args.at(0);
}

template <typename Val>
Result<Val> BuiltinList(const InterpretFuncType<Val>& Interpret,
                        const std::vector<Val>& args) {
  adt::List<Val> l;
  l->reserve(args.size());
  l->assign(args.begin(), args.end());
  return Val{l};
}

template <typename Val>
Result<Val> BuiltinApply(const InterpretFuncType<Val>& Interpret,
                         const std::vector<Val>& args) {
  if (args.size() != 2) {
    return InvalidArgumentError{std::string(kBuiltinId) +
                                "takes 2 arguments, but " +
                                std::to_string(args.size()) + "were given."};
  }
  const auto& pattern_match = ::common::Overloaded{
      [&](const Closure<Val>& closure, const adt::List<Val>& arg_list)
          -> Result<Val> { return Interpret(closure, args_list->vector()); },
      [&](const BuiltinFuncType<Val>& builtin_func,
          const adt::List<Val>& arg_list) -> Result<Val> {
        return builtin_func(Interpret, arg_list->vector());
      },
      [&](const auto&, const auto&) -> Result<Val> {
        if (!args.at(1).template Has<adt::List<Val>>()) {
          return InvalidArgumentError{
              std::string() + "the second arguments must be list, " +
              GetBuiltinTypeName(args.at(1)) + " were given."};
        }
        return InvalidArgumentError{
            std::string() + "the second arguments must be a function, " +
            GetBuiltinTypeName(args.at(0)) + " were given."};
      }};
  return std::visit(pattern_match, args.at(0).varaint(), args.at(1).variant());
}

}  // namespace pexpr
