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

#include "ap/include/axpr/method_class.h"
#include "ap/include/code_gen/in_tensor_data_ptr_kernel_arg_id.h"
#include "ap/include/code_gen/kernel_arg_id_helper.h"

namespace ap::code_gen {

template <typename ValueT, typename BirNode /* background ir node */>
struct InTensorDataPtrKernelArgIdMethodClass {
  using This = InTensorDataPtrKernelArgIdMethodClass;
  using Self = InTensorDataPtrKernelArgId<BirNode>;

  static adt::Result<ValueT> GetAttr(const ValueT& self_val,
                                     const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1);
    const auto& attr_name_val = args.at(0);
    ADT_LET_CONST_REF(attr_name, attr_name_val.template CastTo<std::string>());
    if (attr_name == "type") {
      return This{}.GetArgType(self);
    }
    if (attr_name == "runtime_getter") {
      ADT_CHECK(self->runtime_getter.has_value())
          << adt::errors::ValueError{"no runtime getter initialized"};
      return self->runtime_getter.value();
    }
    return adt::errors::AttributeError{
        std::string() +
        "'InTensorDataPtrKernelArgId' instance has no attribute '" + attr_name +
        "'."};
  }

  adt::Result<ValueT> GetArgType(const Self& self) {
    KernelArgIdHelper<BirNode> helper;
    ADT_LET_CONST_REF(arg_type, helper.GetArgType(self));
    return arg_type.template CastTo<ValueT>();
  }
};

template <typename ValueT, typename BirNode>
const axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>&
GetInTensorDataPtrKernelArgIdClass() {
  using ClassT = axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>;
  using ImplMethods = InTensorDataPtrKernelArgIdMethodClass<ValueT, BirNode>;
  static ClassT cls(axpr::MakeBuiltinClass<ValueT>(
      "InTensorDataPtrKernelArgId", [&](const auto& Define) {
        Define("__getattr__", &ImplMethods::GetAttr);
      }));
  return cls;
}

}  // namespace ap::code_gen
