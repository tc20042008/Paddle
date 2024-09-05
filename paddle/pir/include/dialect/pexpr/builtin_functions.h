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
#include "paddle/pir/include/dialect/pexpr/arithmetic_value_util.h"
#include "paddle/pir/include/dialect/pexpr/value.h"

namespace pexpr {

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

template <typename Op, typename Val>
struct GetBuiltinOpFuncHelper {};

template <typename CustomT>
Result<Value<CustomT>> CustomGetAttr(const CustomT& val,
                                     const std::string& name);

template <typename CustomT>
Result<Value<CustomT>> ValueGetAttr(const Value<CustomT>& val,
                                    const std::string& name) {
  using ValueT = Value<CustomT>;
  return val.Match(
      [&](const Object<ValueT>& obj) -> Result<ValueT> {
        const auto& iter = obj->storage.find(name);
        if (iter == obj->storage.end()) {
          return AttributeError{std::string("no attribute '") + name +
                                "' found."};
        }
        return iter->second;
      },
      [&](const CustomT& custom_val) -> Result<ValueT> {
        return CustomGetAttr(custom_val, name);
      },
      [&](const auto& other) -> Result<ValueT> {
        return AttributeError{std::string("no attribute '") + name +
                              "' found."};
      });
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

#define SPECIALIZE_GetBuiltinOpFuncHelper(op_class, func_template) \
  template <typename Val>                                          \
  struct GetBuiltinOpFuncHelper<op_class, Val> {                   \
    static Val Call() { return &func_template<Val>; }              \
  };

SPECIALIZE_GetBuiltinOpFuncHelper(builtin_symbol::GetAttr, BuiltinGetAttr);

template <typename CustomT>
Result<Value<CustomT>> CustomGetItem(const CustomT& val,
                                     const Value<CustomT>& idx);

template <typename CustomT>
Result<Value<CustomT>> ValueGetItem(const Value<CustomT>& val,
                                    const Value<CustomT>& idx) {
  using ValueT = Value<CustomT>;
  return val.Match(
      [&](const adt::List<ValueT>& obj) -> Result<ValueT> {
        return idx.Match(
            [&](const ArithmeticValue& arithmetic_idx) -> Result<ValueT> {
              const auto& int64_idx =
                  arithmetic_idx.StaticCastTo(CppArithmeticType<int64_t>{});
              ADT_RETURN_IF_ERROR(int64_idx);
              const auto& opt_index =
                  int64_idx.GetOkValue().template TryGet<int64_t>();
              ADT_RETURN_IF_ERROR(opt_index);
              int64_t index = opt_index.GetOkValue();
              if (index < 0) {
                index += obj->size();
              }
              if (index >= 0 && index < obj->size()) {
                return obj->at(index);
              }
              return IndexError{"list index out of range"};
            },
            [&](const auto&) -> Result<ValueT> {
              return TypeError{std::string() +
                               "list indices must be integers, not " +
                               GetBuiltinTypeName(idx)};
            });
      },
      [&](const CustomT& custom_val) -> Result<ValueT> {
        return CustomGetItem(custom_val, idx);
      },
      [&](const auto& other) -> Result<ValueT> {
        return TypeError{std::string() + "'" + GetBuiltinTypeName(val) +
                         "' object is not subscriptable"};
      });
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

SPECIALIZE_GetBuiltinOpFuncHelper(builtin_symbol::GetItem, BuiltinGetItem);

template <typename Val>
Result<Val> BuiltinHalt(const Val&, const std::vector<Val>& args) {
  return RuntimeError{"Dead code. Halt function should never be touched."};
}

template <typename ArithmeticOp, typename Val>
Result<Val> BuiltinArithmeticUnary(const ArithmeticValue& value) {
  const Result<ArithmeticValue>& ret = ArithmeticUnaryFunc<ArithmeticOp>(value);
  ADT_RETURN_IF_ERROR(ret);
  return ret.GetOkValue();
}

template <typename ArithmeticOp, typename Val>
Result<Val> BuiltinUnary(const Val&, const std::vector<Val>& args) {
  if (args.size() != 1) {
    return adt::errors::TypeError{std::string() + "'" + ArithmeticOp::Name() +
                                  "' only support 1 arguments."};
  }
  const Val& value = args.at(0);
  return value.Match(
      [&](const ArithmeticValue& impl) -> Result<Val> {
        return BuiltinArithmeticUnary<ArithmeticOp, Val>(impl);
      },
      [&](const auto&) -> Result<Val> {
        return adt::errors::TypeError{
            std::string() + "'" + GetBuiltinTypeName(value) +
            "' type does not support unary op '" + ArithmeticOp::Name() + "'"};
      });
}

#define SPECIALIZE_GetBuiltinOpFuncHelper_BuiltinUnary(cls_name, op)       \
  template <typename Val>                                                  \
  struct GetBuiltinOpFuncHelper<builtin_symbol::cls_name, Val> {           \
    static Val Call() { return &BuiltinUnary<Arithmetic##cls_name, Val>; } \
  };

PEXPR_FOR_EACH_UNARY_OP(SPECIALIZE_GetBuiltinOpFuncHelper_BuiltinUnary);

#undef SPECIALIZE_GetBuiltinOpFuncHelper_BuiltinUnary

template <typename ArithmeticOp, typename Val>
Result<Val> BuiltinArithmeticBinary(const ArithmeticValue& lhs,
                                    const Val& rhs_val) {
  const Result<ArithmeticValue>& rhs =
      CastToBuiltinValue<ArithmeticValue>(rhs_val);
  ADT_RETURN_IF_ERROR(rhs);
  const auto& ret = ArithmeticBinaryFunc<ArithmeticOp>(lhs, rhs.GetOkValue());
  ADT_RETURN_IF_ERROR(ret);
  return ret.GetOkValue();
}

template <typename ArithmeticOp, typename Val, typename T>
struct BuiltinStringBinaryHelper {
  static Result<Val> Call(const std::string& str, const T& rhs) {
    return adt::errors::TypeError{std::string() +
                                  "unsupported operand types for " +
                                  ArithmeticOp::Name() + ": 'str' and '" +
                                  GetBuiltinTypeNameImpl<T>() + "'"};
  }
};

template <typename Val>
struct BuiltinStringBinaryHelper<ArithmeticAdd, Val, std::string> {
  static Result<Val> Call(const std::string& lhs, const std::string& rhs) {
    return ArithmeticAdd::Call(lhs, rhs);
  }
};

template <typename Val>
struct BuiltinStringBinaryHelper<ArithmeticAdd, Val, ArithmeticValue> {
  static Result<Val> Call(const std::string& lhs,
                          const ArithmeticValue& rhs_val) {
    const auto& rhs = rhs_val.Match([](auto impl) -> Result<std::string> {
      using T = decltype(impl);
      if constexpr (IsArithmeticOpSupported<T>()) {
        return std::to_string(impl);
      } else {
        return adt::errors::TypeError{std::string() +
                                      "unsupported operand types for " +
                                      ArithmeticAdd::Name() + ": 'str' and '" +
                                      CppArithmeticType<T>{}.Name() + "'"};
      }
    });
    ADT_RETURN_IF_ERROR(rhs);
    return lhs + rhs.GetOkValue();
  }
};

#define SPECIALIZE_BuiltinStringBinaryHelper_string_cmp(cls_name)             \
  template <typename Val>                                                     \
  struct BuiltinStringBinaryHelper<cls_name, Val, std::string> {              \
    static Result<Val> Call(const std::string& lhs, const std::string& rhs) { \
      return ArithmeticValue{cls_name::Call(lhs, rhs)};                       \
    }                                                                         \
  };
SPECIALIZE_BuiltinStringBinaryHelper_string_cmp(ArithmeticEQ);
SPECIALIZE_BuiltinStringBinaryHelper_string_cmp(ArithmeticNE);
SPECIALIZE_BuiltinStringBinaryHelper_string_cmp(ArithmeticGT);
SPECIALIZE_BuiltinStringBinaryHelper_string_cmp(ArithmeticGE);
SPECIALIZE_BuiltinStringBinaryHelper_string_cmp(ArithmeticLT);
SPECIALIZE_BuiltinStringBinaryHelper_string_cmp(ArithmeticLE);
#undef SPECIALIZE_BuiltinStringBinaryHelper_string

template <typename Val>
struct BuiltinStringBinaryHelper<ArithmeticMul, Val, ArithmeticValue> {
  static Result<Val> Call(const std::string& lhs,
                          const ArithmeticValue& rhs_val) {
    const auto& opt_uint64 =
        rhs_val.StaticCastTo(CppArithmeticType<uint64_t>{});
    ADT_RETURN_IF_ERROR(opt_uint64);
    const auto& opt_size = opt_uint64.GetOkValue().template TryGet<uint64_t>();
    ADT_RETURN_IF_ERROR(opt_size);
    uint64_t size = opt_size.GetOkValue();
    std::ostringstream ss;
    for (int i = 0; i < size; ++i) {
      ss << lhs;
    }
    return ss.str();
  }
};

template <typename ArithmeticOp, typename Val>
Result<Val> BuiltinStringBinary(const std::string& str, const Val& rhs_val) {
  return rhs_val.Match([&](const auto& rhs) -> Result<Val> {
    using T = std::decay_t<decltype(rhs)>;
    return BuiltinStringBinaryHelper<ArithmeticOp, Val, T>::Call(str, rhs);
  });
}

template <typename ArithmeticOp, typename Val>
Result<Val> BuiltinBinary(const Val&, const std::vector<Val>& args) {
  if (args.size() != 2) {
    return adt::errors::TypeError{std::string() + "'" + ArithmeticOp::Name() +
                                  "' only support 2 arguments."};
  }
  const Val& lhs_val = args.at(0);
  const Val& rhs_val = args.at(1);
  return lhs_val.Match(
      [&](const ArithmeticValue& lhs_impl) -> Result<Val> {
        return BuiltinArithmeticBinary<ArithmeticOp, Val>(lhs_impl, rhs_val);
      },
      [&](const std::string& str) -> Result<Val> {
        return BuiltinStringBinary<ArithmeticOp, Val>(str, rhs_val);
      },
      [&](const auto&) -> Result<Val> {
        return adt::errors::TypeError{
            std::string() + "'" + GetBuiltinTypeName(lhs_val) +
            "' type does not support binary op '" + ArithmeticOp::Name() + "'"};
      });
}

#define SPECIALIZE_GetBuiltinOpFuncHelper_BuiltinBinary(cls_name, op)       \
  template <typename Val>                                                   \
  struct GetBuiltinOpFuncHelper<builtin_symbol::cls_name, Val> {            \
    static Val Call() { return &BuiltinBinary<Arithmetic##cls_name, Val>; } \
  };

PEXPR_FOR_EACH_BINARY_OP(SPECIALIZE_GetBuiltinOpFuncHelper_BuiltinBinary);

#undef SPECIALIZE_GetBuiltinOpFuncHelper_BuiltinBinary

template <typename Op, typename Val>
Val GetBuiltinOpFunc() {
  return GetBuiltinOpFuncHelper<Op, Val>::Call();
}

}  // namespace pexpr
