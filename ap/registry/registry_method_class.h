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

#include "ap/axpr/method.h"
#include "ap/axpr/method_class.h"
#include "ap/axpr/type.h"
#include "ap/registry/registry.h"
#include "ap/registry/registry_keys.h"
#include "ap/registry/registry_singleton.h"
#include "ap/registry/setter_decorator.h"

namespace ap::registry {

template <typename ValueT>
struct RegistryMethodClass {};

template <typename ValueT>
struct TypeImplRegistryMethodClass {
  using This = TypeImplRegistryMethodClass;
  using Self = axpr::TypeImpl<Registry<ValueT>>;

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, axpr::TryGetImpl<std::string>(attr_name_val));
    if (attr_name == kOpIndexExpr()) {
      return axpr::Method<ValueT>{self, &This::RegisterOpIndexExpr};
    }
    if (attr_name == kDrr()) {
      return axpr::Method<ValueT>{self, &This::RegisterDrr};
    }
    if (attr_name == kOpCompute()) {
      return axpr::Method<ValueT>{self, &This::RegisterOpCompute};
    }
    return adt::errors::AttributeError{std::string() +
                                       "'Registry' object has no attribute '" +
                                       attr_name + "'"};
  }

  static adt::Result<ValueT> RegisterOpIndexExpr(
      const ValueT& self_val, const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 2)
        << adt::errors::TypeError{std::string() + "'Registry." +
                                  kOpIndexExpr() + "' takes 2 arguments. but " +
                                  std::to_string(args.size()) + " were given."};
    const auto& op_names_val = args.at(0);
    ADT_LET_CONST_REF(op_names_list,
                      axpr::TryGetImpl<adt::List<ValueT>>(op_names_val))
        << adt::errors::TypeError{std::string() + "argument 1 of 'Registry." +
                                  kOpIndexExpr() + "' should be list, but '" +
                                  axpr::GetTypeName(op_names_val) +
                                  "' were given."};
    std::vector<std::string> op_names;
    for (const auto& elt : *op_names_list) {
      ADT_LET_CONST_REF(op_name, axpr::TryGetImpl<std::string>(elt))
          << adt::errors::TypeError{
                 std::string() + "argument 1 of 'Registry." + kOpIndexExpr() +
                 "' should be list of string, but one item of type '" +
                 axpr::GetTypeName(elt) + "' were found."};
      op_names.emplace_back(op_name);
    }
    const auto& nice_val = args.at(1);
    ADT_LET_CONST_REF(nice, axpr::TryGetImpl<int64_t>(nice_val))
        << adt::errors::TypeError{std::string() + "argument 2 of 'Registry." +
                                  kOpIndexExpr() + "' should be int, but '" +
                                  axpr::GetTypeName(nice_val) +
                                  "' were given."};
    Cell<axpr::Lambda<axpr::CoreExpr>> lambda{};
    for (const auto& op_name : op_names) {
      OpIndexesExprRegistryItem item{op_name, nice, lambda};
      RegistrySingleton<ValueT>::Add(item);
    }
    return SetterDecorator{lambda};
  }

  static adt::Result<ValueT> RegisterDrr(const ValueT& self_val,
                                         const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 2) << adt::errors::TypeError{
        std::string() + "'Registry." + kDrr() + "' takes 2 arguments. but " +
        std::to_string(args.size()) + " were given."};
    const auto& drr_name_val = args.at(0);
    ADT_LET_CONST_REF(drr_name, axpr::TryGetImpl<std::string>(drr_name_val))
        << adt::errors::TypeError{std::string() + "argument 1 of 'Registry." +
                                  kDrr() + "' should be string, but '" +
                                  axpr::GetTypeName(drr_name_val) +
                                  "' were given."};
    const auto& nice_val = args.at(1);
    ADT_LET_CONST_REF(nice, axpr::TryGetImpl<int64_t>(nice_val))
        << adt::errors::TypeError{std::string() + "argument 2 of 'Registry." +
                                  kDrr() + "' should be int, but '" +
                                  axpr::GetTypeName(nice_val) +
                                  "' were given."};
    Cell<axpr::Lambda<axpr::CoreExpr>> lambda{};
    DrrRegistryItem item{drr_name, nice, lambda};
    RegistrySingleton<ValueT>::Add(item);
    return SetterDecorator{lambda};
  }

  static adt::Result<ValueT> RegisterOpCompute(
      const ValueT& self_val, const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 3)
        << adt::errors::TypeError{std::string() + "'Registry." + kOpCompute() +
                                  "' takes 3 arguments. but " +
                                  std::to_string(args.size()) + " were given."};
    const auto& op_name_val = args.at(0);
    ADT_LET_CONST_REF(op_name, axpr::TryGetImpl<std::string>(op_name_val))
        << adt::errors::TypeError{std::string() + "argument 1 of 'Registry." +
                                  kOpCompute() + "' should be string, but '" +
                                  axpr::GetTypeName(op_name_val) +
                                  "' were given."};
    const auto& arch_type_val = args.at(1);
    ADT_LET_CONST_REF(arch_type, axpr::TryGetImpl<std::string>(arch_type_val))
        << adt::errors::TypeError{std::string() + "argument 2 of 'Registry." +
                                  kOpCompute() + "' should be int, but '" +
                                  axpr::GetTypeName(arch_type_val) +
                                  "' were given."};
    const auto& nice_val = args.at(2);
    ADT_LET_CONST_REF(nice, axpr::TryGetImpl<int64_t>(nice_val))
        << adt::errors::TypeError{std::string() + "argument 2 of 'Registry." +
                                  kOpCompute() + "' should be int, but '" +
                                  axpr::GetTypeName(nice_val) +
                                  "' were given."};
    Cell<axpr::Lambda<axpr::CoreExpr>> lambda{};
    OpComputeRegistryItem item{op_name, arch_type, nice, lambda};
    RegistrySingleton<ValueT>::Add(item);
    return SetterDecorator{lambda};
  }
};

}  // namespace ap::registry

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, registry::Registry<ValueT>>
    : public registry::RegistryMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<registry::Registry<ValueT>>>
    : public registry::TypeImplRegistryMethodClass<ValueT> {};

}  // namespace ap::axpr
