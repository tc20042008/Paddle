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

#include "ap/axpr/method_class.h"
#include "ap/code_gen/code_gen_ctx.h"
#include "ap/code_gen/dim_expr_kernel_arg_id.h"
#include "ap/code_gen/kernel_arg_id_helper.h"

namespace ap::code_gen {

template <typename ValueT, typename BirNode /* background ir node */>
struct DimExprKernelArgIdMethodClass {
  using This = DimExprKernelArgIdMethodClass;
  using Self = DimExprKernelArgId<BirNode>;

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    if (attr_name == "value") {
      return self->template CastData<ValueT>();
    }
    if (attr_name == "type") {
      return GetArgType(self);
    }
    return adt::errors::AttributeError{
        std::string() + "'DimExprKernelArgId' instance has no attribute '" +
        attr_name + "'."};
  }

  adt::Result<ValueT> GetArgType(const Self& self) {
    KernelArgIdHelper<BirNode> helper;
    ADT_LET_CONST_REF(arg_type, helper.GetArgType(self));
    return arg_type.template CastTo<ValueT>();
  }
};

template <typename ValueT, typename BirNode /* background ir node */>
struct TypeImplDimExprKernelArgIdMethodClass {
  using This = TypeImplDimExprKernelArgIdMethodClass;
  using Self = axpr::TypeImpl<DimExprKernelArgId<BirNode>>;
};

}  // namespace ap::code_gen

namespace ap::axpr {

template <typename ValueT, typename BirNode /* background ir node */>
struct MethodClassImpl<ValueT, code_gen::DimExprKernelArgId<BirNode>>
    : public code_gen::DimExprKernelArgIdMethodClass<ValueT, BirNode> {};

template <typename ValueT, typename BirNode /* background ir node */>
struct MethodClassImpl<ValueT, TypeImpl<code_gen::DimExprKernelArgId<BirNode>>>
    : public code_gen::TypeImplDimExprKernelArgIdMethodClass<ValueT, BirNode> {
};

}  // namespace ap::axpr
