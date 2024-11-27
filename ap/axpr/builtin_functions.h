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
#include <sstream>
#include "ap/axpr/abstract_list.h"
#include "ap/axpr/bool_int_double_helper.h"
#include "ap/axpr/builtin_high_order_func_type.h"
#include "ap/axpr/data_value_util.h"
#include "ap/axpr/method_class.h"
#include "ap/axpr/string_util.h"
#include "ap/axpr/value.h"

namespace ap::axpr {

namespace detail {

template <typename Val>
adt::Result<bool> ConvertToBool(const Val& cond) {
  using TypeT = typename TypeTrait<Val>::TypeT;
  return cond.Match(
      [](const TypeT&) -> Result<bool> { return true; },
      [](const bool c) -> Result<bool> { return c; },
      [](const int64_t c) -> Result<bool> { return c != 0; },
      [](const double c) -> Result<bool> { return c != 0; },
      [](const std::string& c) -> Result<bool> { return !c.empty(); },
      [](const Nothing&) -> Result<bool> { return false; },
      [](const adt::List<Val>& list) -> Result<bool> {
        return list->size() > 0;
      },
      [](const MutableList<Val>& list) -> Result<bool> {
        ADT_LET_CONST_REF(list_ptr, list.Get());
        return list_ptr->size() > 0;
      },
      [](const BuiltinObject<Val>& obj) -> Result<bool> {
        return obj->size() > 0;
      },
      [](const Lambda<CoreExpr>&) -> Result<bool> { return true; },
      [](const Closure<Val>&) -> Result<bool> { return true; },
      [](const Continuation<Val>&) -> Result<bool> { return true; },
      [](const Method<Val>&) -> Result<bool> { return true; },
      [](const builtin_symbol::Symbol&) -> Result<bool> { return true; },
      [](const BuiltinFuncType<Val>&) -> Result<bool> { return true; },
      [](const BuiltinHighOrderFuncType<Val>&) -> Result<bool> { return true; },
      [](const CpsBuiltinHighOrderFuncType<Val>&) -> Result<bool> {
        return true;
      },
      [&](const auto&) -> Result<bool> {
        return TypeError{std::string() + "'" + axpr::GetTypeName(cond) +
                         "' could not be convert to bool"};
      });
}

}  // namespace detail

template <typename Val>
Result<adt::Ok> CpsBuiltinIf(InterpreterBase<Val>* interpreter,
                             ComposedCallImpl<Val>* composed_call) {
  const auto args = composed_call->args;
  if (args.size() != 3) {
    return TypeError{std::string("`if` takes 3 arguments, but ") +
                     std::to_string(args.size()) + "were given."};
  }
  const auto& cond = args.at(0);
  ADT_LET_CONST_REF(select_true_branch, detail::ConvertToBool<Val>(cond));
  const auto& opt_true_closure = args.at(1).template TryGet<Closure<Val>>();
  ADT_RETURN_IF_ERR(opt_true_closure);
  const auto& true_closure = opt_true_closure.GetOkValue();
  const auto& opt_false_closure = args.at(2).template TryGet<Closure<Val>>();
  ADT_RETURN_IF_ERR(opt_false_closure);
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
  const auto& opt_arg_list = args.at(1).template TryGet<adt::List<Val>>();
  if (!opt_arg_list.HasOkValue()) {
    return TypeError{std::string() + "the second arguments must be list, " +
                     axpr::GetTypeName(args.at(1)) + " were given."};
  }
  const auto& arg_list = opt_arg_list.GetOkValue();
  composed_call->inner_func = args.at(0);
  composed_call->args = arg_list.vector();
  return adt::Ok{};
}

template <typename Val>
Result<Val> BuiltinIdentity(const Val&, const std::vector<Val>& args) {
  if (args.size() != 1) {
    return TypeError{std::string(kBuiltinIdentity()) +
                     "takes 1 argument, but " + std::to_string(args.size()) +
                     "were given."};
  }
  return args.at(0);
}

template <typename ValueT>
Result<ValueT> BuiltinList(const ValueT&, const std::vector<ValueT>& args) {
  adt::List<ValueT> l;
  for (const auto& arg : args) {
    const auto& arg_ret = arg.Match(
        [&](const Starred<ValueT>& starred) -> Result<adt::Ok> {
          ADT_LET_CONST_REF(sublist,
                            starred->obj.template TryGet<adt::List<ValueT>>());
          for (const auto& elt : *sublist) {
            l->emplace_back(elt);
          }
          return adt::Ok{};
        },
        [&](const auto&) -> Result<adt::Ok> {
          l->emplace_back(arg);
          return adt::Ok{};
        });
    ADT_RETURN_IF_ERR(arg_ret);
  }
  return ValueT{l};
}

template <typename Val>
Result<Val> BuiltinHalt(const Val&, const std::vector<Val>& args) {
  return RuntimeError{"Dead code. Halt function should never be touched."};
}

template <typename ValueT>
adt::Result<ValueT> Print(const ValueT&, const std::vector<ValueT>& args) {
  std::ostringstream ss;
  int i = 0;
  for (const auto& obj : args) {
    if (i++ > 0) {
      ss << " ";
    }
    const auto& func = MethodClass<ValueT>::ToString(obj);
    ADT_LET_CONST_REF(str_val, func(obj));
    ADT_LET_CONST_REF(str, str_val.template TryGet<std::string>())
        << adt::errors::TypeError{
               std::string() + "'" + axpr::GetTypeName(obj) +
               ".__builtin_ToString__ should return a 'str' but '" +
               axpr::GetTypeName(str_val) + "' were returned."};
    ss << str;
  }
  LOG(ERROR) << "Print\n" << ss.str();
  return adt::Nothing{};
}

template <typename ValueT>
adt::Result<ValueT> ReplaceOrTrimLeftComma(const ValueT&,
                                           const std::vector<ValueT>& args) {
  ADT_CHECK(args.size() == 3) << adt::errors::TypeError{
      std::string() + "'replace_or_trim_left_comma' takes 3 arguments but " +
      std::to_string(args.size()) + " were given."};
  ADT_LET_CONST_REF(self, args.at(0).template TryGet<std::string>())
      << adt::errors::TypeError{
             std::string() +
             "the argument 1 of 'replace_or_trim_left_comma' should be a str "
             "(not '" +
             axpr::GetTypeName(args.at(0)) + "')."};
  ADT_LET_CONST_REF(pattern, args.at(1).template TryGet<std::string>())
      << adt::errors::TypeError{
             std::string() +
             "the argument 2 of 'replace_or_trim_left_comma' should be a str "
             "(not '" +
             axpr::GetTypeName(args.at(1)) + "')."};
  ADT_LET_CONST_REF(replacement, args.at(2).template TryGet<std::string>())
      << adt::errors::TypeError{
             std::string() +
             "the argument 3 of 'replace_or_trim_left_comma' should be a str "
             "(not '" +
             axpr::GetTypeName(args.at(2)) + "')."};
  std::size_t pattern_pos = self.find(pattern);
  if (pattern_pos == std::string::npos) {
    return self;
  }
  auto EquivalentComma =
      [](const std::string& self, std::size_t start, std::size_t end) {
        if (start == std::string::npos) {
          return false;
        }
        if (start < 0) {
          return false;
        }
        if (start >= self.size()) {
          return false;
        }
        if (end == std::string::npos) {
          return false;
        }
        if (end < 0) {
          return false;
        }
        if (end >= self.size()) {
          return false;
        }
        if (start >= end) {
          return false;
        }
        if (self[start] != ',') {
          return false;
        }
        for (int i = start + 1; i < end; ++i) {
          char ch = self[i];
          if (ch == ' ') {
            continue;
          }
          if (ch == '\r') {
            continue;
          }
          if (ch == '\n') {
            continue;
          }
          if (ch == '\t') {
            continue;
          }
          return false;
        }
        return true;
      };
  if (replacement.empty()) {
    std::size_t comma_pos = self.rfind(',', pattern_pos);
    if (EquivalentComma(self, comma_pos, pattern_pos)) {
      std::string str = self;
      return str.replace(comma_pos, pattern_pos + pattern.size(), "");
    } else {
      return self;
    }
  } else {
    std::string str = self;
    return str.replace(pattern_pos, pattern.size(), replacement);
  }
}

template <typename ValueT>
adt::Result<ValueT> MakeRange(const ValueT&, const std::vector<ValueT>& args) {
  std::optional<int64_t> start;
  std::optional<int64_t> end;
  if (args.size() == 1) {
    start = 0;
    ADT_LET_CONST_REF(arg0, args.at(0).template TryGet<int64_t>())
        << adt::errors::TypeError{
               std::string() + "'range' takes int argument but " +
               axpr::GetTypeName(args.at(0)) + " were given."};
    end = arg0;
  } else if (args.size() == 2) {
    ADT_LET_CONST_REF(arg0, args.at(0).template TryGet<int64_t>())
        << adt::errors::TypeError{
               std::string() + "'range' takes int argument but " +
               axpr::GetTypeName(args.at(0)) + " were given."};
    ADT_LET_CONST_REF(arg1, args.at(1).template TryGet<int64_t>())
        << adt::errors::TypeError{
               std::string() + "'range' takes int argument but " +
               axpr::GetTypeName(args.at(1)) + " were given."};
    start = arg0;
    end = arg1;
  } else {
    ADT_CHECK(false) << adt::errors::TypeError{
        std::string() + "'range' takes 1 or 2 arguments but " +
        std::to_string(args.size()) + " were given."};
  }
  ADT_CHECK(start.has_value());
  ADT_CHECK(end.has_value());
  adt::List<ValueT> ret;
  ret->reserve((start.value() > end.value() ? 0 : end.value() - start.value()));
  for (int64_t i = start.value(); i < end.value(); ++i) {
    ret->emplace_back(i);
  }
  return ret;
}

template <typename Val>
Result<Val> Map(axpr::InterpreterBase<Val>* interpreter,
                const Val&,
                const std::vector<Val>& args) {
  ADT_CHECK(args.size() == 2)
      << adt::errors::TypeError{std::string() + "map() takes 2 arguments but " +
                                std::to_string(args.size()) + " were given."};

  ADT_LET_CONST_REF(lst, axpr::AbstractList<Val>::CastFrom(args.at(1)));
  ADT_LET_CONST_REF(lst_size, lst.size());
  adt::List<Val> ret;
  ret->reserve(lst_size);
  const auto& f = args.at(0);
  ADT_RETURN_IF_ERR(
      lst.Visit([&](const auto& elt) -> adt::Result<adt::LoopCtrl> {
        ADT_LET_CONST_REF(converted_elt,
                          interpreter->InterpretCall(f, std::vector<Val>{elt}));
        ret->emplace_back(converted_elt);
        return adt::Continue{};
      }));
  return ret;
}

template <typename Val>
Result<Val> Filter(axpr::InterpreterBase<Val>* interpreter,
                   const Val&,
                   const std::vector<Val>& args) {
  ADT_CHECK(args.size() == 2) << adt::errors::TypeError{
      std::string() + "filter() takes 2 arguments but " +
      std::to_string(args.size()) + " were given."};

  ADT_LET_CONST_REF(lst, axpr::AbstractList<Val>::CastFrom(args.at(1)));
  ADT_LET_CONST_REF(lst_size, lst.size());
  adt::List<Val> ret;
  ret->reserve(lst_size);
  const auto& f = args.at(0);
  ADT_RETURN_IF_ERR(
      lst.Visit([&](const auto& elt) -> adt::Result<adt::LoopCtrl> {
        ADT_LET_CONST_REF(filter_result,
                          interpreter->InterpretCall(f, std::vector<Val>{elt}));
        ADT_LET_CONST_REF(is_true, detail::ConvertToBool<Val>(filter_result));
        if (is_true) {
          ret->emplace_back(elt);
        }
        return adt::Continue{};
      }));
  return ret;
}

template <typename Val>
Result<Val> Zip(const Val&, const std::vector<Val>& args) {
  std::optional<std::size_t> size;
  for (const auto& arg : args) {
    ADT_LET_CONST_REF(lst, axpr::AbstractList<Val>::CastFrom(arg))
        << adt::errors::TypeError{std::string() +
                                  "the argument of 'zip' should be list."};
    ADT_LET_CONST_REF(lst_size, lst.size());
    if (size.has_value()) {
      ADT_CHECK(size.value() == lst_size) << adt::errors::TypeError{
          std::string() + "the arguments of 'zip' should be the same size."};
    } else {
      size = lst_size;
    }
  }
  adt::List<Val> ret;
  ret->reserve(size.value());
  for (int i = 0; i < size.value(); ++i) {
    adt::List<Val> tuple;
    tuple->reserve(args.size());
    for (const auto& arg : args) {
      ADT_LET_CONST_REF(lst, axpr::AbstractList<Val>::CastFrom(arg));
      ADT_LET_CONST_REF(elt, lst.at(i));
      tuple->emplace_back(elt);
    }
    ret->emplace_back(tuple);
  }
  return ret;
}

template <typename Val>
Result<Val> Reduce(axpr::InterpreterBase<Val>* interpreter,
                   const Val&,
                   const std::vector<Val>& args) {
  ADT_CHECK(args.size() == 2 || args.size() == 3) << adt::errors::TypeError{
      std::string() + "'reduce' takes 2 or 3 arguments but " +
      std::to_string(args.size()) + " were given."};
  ADT_LET_CONST_REF(lst, axpr::AbstractList<Val>::CastFrom(args.at(1)));
  std::optional<Val> init;
  std::optional<int64_t> start;
  ADT_LET_CONST_REF(lst_size, lst.size());
  if (lst_size > 0) {
    ADT_LET_CONST_REF(init_val, lst.at(0));
    init = init_val;
    start = 1;
  } else {
    ADT_CHECK(args.size() == 3) << adt::errors::TypeError{
        std::string() + "reduce() of empty sequence with no initial value"};
    init = args.at(2);
    start = 0;
  }
  ADT_CHECK(init.has_value());
  ADT_CHECK(start.has_value());
  Val ret{init.value()};
  const auto& f = args.at(0);
  for (int i = start.value(); i < lst_size; ++i) {
    ADT_LET_CONST_REF(elt, lst.at(i));
    ADT_LET_CONST_REF(
        cur_reduced, interpreter->InterpretCall(f, std::vector<Val>{elt, ret}));
    ret = cur_reduced;
  }
  return ret;
}

template <typename Val>
Result<Val> Max(const Val&, const std::vector<Val>& args) {
  ADT_CHECK(args.size() == 2)
      << adt::errors::TypeError{std::string() + "max() takes 2 arguments but " +
                                std::to_string(args.size()) + " were given."};
  ADT_LET_CONST_REF(lhs, BoolIntDouble::CastFrom(args.at(0)))
      << adt::errors::TypeError{std::string() +
                                "the argument 1 of max() should be 'bool', "
                                "'int' or 'float' (not '" +
                                axpr::GetTypeName(args.at(0)) + "')."};
  ADT_LET_CONST_REF(rhs, BoolIntDouble::CastFrom(args.at(1)))
      << adt::errors::TypeError{std::string() +
                                "the argument 1 of max() should be 'bool', "
                                "'int' or 'float' (not '" +
                                axpr::GetTypeName(args.at(0)) + "')."};
  BoolIntDoubleHelper<Val> helper{};
  ADT_LET_CONST_REF(cmp_ret,
                    helper.template BinaryFunc<ArithmeticGE>(lhs, rhs));
  ADT_LET_CONST_REF(cmp, cmp_ret.template TryGet<bool>());
  return cmp ? args.at(0) : args.at(1);
}

template <typename Val>
Result<Val> Min(const Val&, const std::vector<Val>& args) {
  ADT_CHECK(args.size() == 2)
      << adt::errors::TypeError{std::string() + "min() takes 2 arguments but " +
                                std::to_string(args.size()) + " were given."};
  ADT_LET_CONST_REF(lhs, BoolIntDouble::CastFrom(args.at(0)))
      << adt::errors::TypeError{std::string() +
                                "the argument 1 of min() should be 'bool', "
                                "'int' or 'float' (not '" +
                                axpr::GetTypeName(args.at(0)) + "')."};
  ADT_LET_CONST_REF(rhs, BoolIntDouble::CastFrom(args.at(1)))
      << adt::errors::TypeError{std::string() +
                                "the argument 1 of min() should be 'bool', "
                                "'int' or 'float' (not '" +
                                axpr::GetTypeName(args.at(0)) + "')."};
  BoolIntDoubleHelper<Val> helper{};
  ADT_LET_CONST_REF(cmp_ret,
                    helper.template BinaryFunc<ArithmeticLE>(lhs, rhs));
  ADT_LET_CONST_REF(cmp, cmp_ret.template TryGet<bool>());
  return cmp ? args.at(0) : args.at(1);
}

}  // namespace ap::axpr
