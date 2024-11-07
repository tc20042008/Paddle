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
#include "ap/kernel_define/in_tensor_data_ptr_kernel_arg_id.h"
#include "ap/kernel_define/kernel_arg_id_helper.h"

namespace ap::kernel_define {

template <typename ValueT, typename BirNode /* background ir node */>
struct OutTensorDataPtrKernelArgIdMethodClass {
  using This = OutTensorDataPtrKernelArgIdMethodClass;
  using Self = OutTensorDataPtrKernelArgId<BirNode>;

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    if (attr_name == "value") {
      return self->template CastData<ValueT>();
    }
    if (attr_name == "type") {
      return GetArgType(self);
    }
    return adt::errors::AttributeError{
        std::string() +
        "'OutTensorDataPtrKernelArgId' instance has no attribute '" +
        attr_name + "'."};
  }

  adt::Result<ValueT> GetArgType(const Self& self) {
    KernelArgIdHelper<BirNode> helper;
    ADT_LET_CONST_REF(arg_type, helper.GetArgType(self));
    return arg_type.template CastTo<ValueT>();
  }
};

template <typename ValueT, typename BirNode /* background ir node */>
struct TypeImplOutTensorDataPtrKernelArgIdMethodClass {
  using This = TypeImplOutTensorDataPtrKernelArgIdMethodClass;
  using Self = axpr::TypeImpl<OutTensorDataPtrKernelArgId<BirNode>>;
};

}  // namespace ap::kernel_define

namespace ap::axpr {

template <typename ValueT, typename BirNode /* background ir node */>
struct MethodClassImpl<ValueT,
                       kernel_define::OutTensorDataPtrKernelArgId<BirNode>>
    : public kernel_define::OutTensorDataPtrKernelArgIdMethodClass<ValueT,
                                                                   BirNode> {};

template <typename ValueT, typename BirNode /* background ir node */>
struct MethodClassImpl<
    ValueT,
    TypeImpl<kernel_define::OutTensorDataPtrKernelArgId<BirNode>>>
    : public kernel_define::
          TypeImplOutTensorDataPtrKernelArgIdMethodClass<ValueT, BirNode> {};

}  // namespace ap::axpr
