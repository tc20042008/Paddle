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

#include <string>
#include "ap/axpr/constants.h"
#include "ap/axpr/method_class.h"
#include "ap/axpr/string_util.h"

namespace ap::axpr {

template <typename ValueT>
struct StringMethodClass {
  using This = StringMethodClass;
  using Self = std::string;

  adt::Result<ValueT> ToString(const Self& self) { return self; }

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    if (attr_name == "replace") {
      return axpr::Method<ValueT>{self, &This::StaticReplace};
    }
    return adt::errors::TypeError{};
  }

  static adt::Result<ValueT> StaticReplace(const ValueT& self_val,
                                           const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template TryGet<Self>());
    ADT_CHECK(args.size() == 2) << adt::errors::TypeError{
        std::string() + "'str.replace' takes 2 arguments but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(pattern, args.at(0).template TryGet<std::string>())
        << adt::errors::TypeError{
               std::string() +
               "the argument 1 of 'str.replace' should be a str"};
    ADT_LET_CONST_REF(replacement, args.at(1).template TryGet<std::string>())
        << adt::errors::TypeError{
               std::string() +
               "the argument 2 of 'str.replace' should be a str"};
    return This{}.Replace(self, pattern, replacement);
  }

  std::string Replace(std::string self,
                      const std::string& pattern,
                      const std::string& replacement) {
    std::size_t pos = self.find(pattern);
    if (pos == std::string::npos) {
      return self;
    }
    return self.replace(pos, pattern.size(), replacement);
  }

  template <typename BultinBinarySymbol>
  static std::optional<BuiltinBinaryFuncT<ValueT>> GetBuiltinBinaryFunc() {
    if constexpr (ConvertBuiltinSymbolToArithmetic<
                      BultinBinarySymbol>::convertable) {
      using ArithmeticOp = typename ConvertBuiltinSymbolToArithmetic<
          BultinBinarySymbol>::arithmetic_op_type;
      return &This::template BinaryFunc<ArithmeticOp>;
    } else {
      return std::nullopt;
    }
  }

  template <typename ArithmeticOp>
  static adt::Result<ValueT> BinaryFunc(const ValueT& lhs_val,
                                        const ValueT& rhs_val) {
    const auto& opt_lhs =
        MethodClass<ValueT>::template TryGet<std::string>(lhs_val);
    ADT_RETURN_IF_ERR(opt_lhs);
    const auto& lhs = opt_lhs.GetOkValue();
    return BuiltinStringBinary<ArithmeticOp>(lhs, rhs_val);
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, std::string> : public StringMethodClass<ValueT> {
};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<std::string>>
    : public EmptyMethodClass<ValueT> {};

}  // namespace ap::axpr
