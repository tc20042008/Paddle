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
#include "ap/axpr/type.h"

#include "ap/drr/native_ir_op_declare.h"
#include "ap/drr/op_pattern_ctx.h"
#include "ap/drr/op_tensor_pattern_ctx_helper.h"
#include "ap/drr/packed_ir_op_declare.h"
#include "ap/drr/res_ptn_packed_ir_op_declare_data.h"
#include "ap/drr/tags.h"
#include "ap/drr/unbound_native_ir_op.h"
#include "ap/drr/unbound_packed_ir_op.h"

namespace ap::drr {

template <typename ValueT, typename NodeT>
struct ResPtnOpPatternCtxMethodClass {
  using This = ResPtnOpPatternCtxMethodClass;
  using ObjT = tResPtn<OpPatternCtx<ValueT, NodeT>>;
  using Self = ObjT;
  using Helper = OpTensorPatternCtxHelper<ValueT, NodeT>;

  adt::Result<ValueT> SetAttr(const Self& self, const ValueT& arg) {
    ADT_LET_CONST_REF(attr_name, axpr::TryGetImpl<std::string>(arg));
    if (IsBasicAttrName(attr_name)) {
      return adt::errors::AttributeError{"can't set attribute '" + attr_name +
                                         "'"};
    } else {
      return axpr::Method<ValueT>{self, &This::MakeAndRegisterUnboundIrOp};
    }
  }

  static adt::Result<ValueT> MakeAndRegisterUnboundIrOp(
      const ValueT& self_val, const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, axpr::TryGetImpl<Self>(self_val));
    ADT_CHECK(args.size() == 2);
    ADT_LET_CONST_REF(op_uid, axpr::TryGetImpl<std::string>(args.at(0)));
    bool has_ir_op = Helper{}.HasIrOpByUid(self.value(), op_uid);
    ADT_CHECK(!has_ir_op) << adt::errors::TypeError{
        std::string() + "op name '" + op_uid +
        "' has been bound. please  bound to a new name."};
    using IrOpT = IrOp<ValueT, NodeT>;
    const auto& opt_ir_op = args.at(1).Match(
        [&](const tResPtn<PackedIrOpDeclare<ValueT, NodeT>>& op)
            -> adt::Result<IrOpT> {
          return UnboundPackedIrOp<ValueT, NodeT>{op.value(), op_uid};
        },
        [&](const tResPtn<NativeIrOpDeclare<ValueT, NodeT>>& op)
            -> adt::Result<IrOpT> {
          return UnboundNativeIrOp<ValueT, NodeT>{op.value(), op_uid};
        },
        [&](const auto&) -> adt::Result<IrOpT> {
          return adt::errors::TypeError{
              std::string() +
              "only 'ResPtnPackedIrOpDeclare' and 'ResPtnNativeIrOpDeclare' "
              "supported for op name binding. '" +
              axpr::GetTypeName(args.at(1)) + "' were given."};
        });
    ADT_LET_CONST_REF(ir_op, opt_ir_op);
    Helper{}.SetIrOpByUid(self.value(), op_uid, ir_op);
    return adt::Nothing{};
  }

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& arg) {
    ADT_LET_CONST_REF(attr_name, axpr::TryGetImpl<std::string>(arg));
    if (IsBasicAttrName(attr_name)) {
      ADT_LET_CONST_REF(attr_getter, FindAttrGetter(attr_name));
      return (this->*attr_getter)(self);
    } else {
      ADT_LET_CONST_REF(ir_op, Helper{}.GetIrOpByUid(self.value(), attr_name));
      return ir_op.Match(
          [](const NativeIrOp<ValueT, NodeT>& impl) -> ValueT { return impl; },
          [](const PackedIrOp<ValueT, NodeT>& impl) -> ValueT { return impl; },
          [](const UnboundNativeIrOp<ValueT, NodeT>& x) -> ValueT {
            return ResPtn(x);
          },
          [](const UnboundPackedIrOp<ValueT, NodeT>& x) -> ValueT {
            return ResPtn(x);
          });
    }
  }

  adt::Result<ValueT> ApPatternFusionOp(const Self& self) {
    return axpr::Method<ValueT>{self, &This::StaticDeclareApPatternFusionOp};
  }

  static adt::Result<ValueT> StaticDeclareApPatternFusionOp(
      const ValueT& self_val, const std::vector<ValueT>& args) {
    return This{}.DeclareApPatternFusionOp(self_val, args);
  }

  adt::Result<ValueT> DeclareApPatternFusionOp(
      const ValueT& self_val, const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, axpr::TryGetImpl<Self>(self_val));
    ADT_CHECK(args.size() == 2) << adt::errors::TypeError{
        std::string() +
        "ResPtnOpPatternCtx.ap_pattern_fusion_op takes 2 arguments. but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(kernel_define_lambda, GetLambdaOfClosure(args.at(0)))
        << adt::errors::TypeError{
               std::string() +
               "argument 1 of o.ap_pattern_fusion_op should be a closure"};
    ADT_LET_CONST_REF(kernel_dispatch_lambda, GetLambdaOfClosure(args.at(1)))
        << adt::errors::TypeError{
               std::string() +
               "argument 2 of o.ap_pattern_fusion_op should be a closure"};
    auto data = std::make_shared<ResPtnPackedIrOpDeclareData>(
        kernel_define_lambda, kernel_dispatch_lambda);
    PackedIrOpDeclare<ValueT, NodeT> op_declare{
        "ap_pattern_fusion_op", self.value().shared_ptr(), data};
    return ResPtn(op_declare);
  }

  adt::Result<axpr::Lambda<axpr::CoreExpr>> GetLambdaOfClosure(
      const ValueT& val) {
    ADT_LET_CONST_REF(closure, val.template TryGet<axpr::Closure<ValueT>>());
    return closure->lambda;
  }

  adt::Result<ValueT> ApNativeOp(const Self& self) {
    return axpr::Method<ValueT>{self, &This::DeclareNativeIrOp};
  }

  static adt::Result<ValueT> DeclareNativeIrOp(
      const ValueT& self_val, const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, axpr::TryGetImpl<Self>(self_val));
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() +
        "ResPtnOpPatternCtx.ap_native_op takes 1 arguments. but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(op_name, axpr::TryGetImpl<std::string>(args.at(0)));
    NativeIrOpDeclare<ValueT, NodeT> op_declare{op_name,
                                                self.value().shared_ptr()};
    return ResPtn(op_declare);
  }

  using AttrGetter = adt::Result<ValueT> (This::*)(const Self&);
  adt::Result<AttrGetter> FindAttrGetter(const std::string& attr_name) {
    const auto& attr_getters = AttrGetters();
    const auto& iter = attr_getters.find(attr_name);
    if (iter == attr_getters.end()) {
      return adt::errors::AttributeError{
          std::string() + "'" + axpr::TypeImpl<ObjT>{}.Name() +
          "' object has no attribute '" + attr_name + "'"};
    }
    return iter->second;
  }

  bool IsBasicAttrName(const std::string& attr_name) {
    const auto& attr_getters = AttrGetters();
    return attr_getters.count(attr_name) > 0;
  }

  const std::map<std::string, AttrGetter>& AttrGetters() {
    static const std::map<std::string, AttrGetter> map{
        {"ap_pattern_fusion_op", &This::ApPatternFusionOp},
        // {"ap_native_op", &This::ApNativeOp},
    };
    return map;
  }
};

}  // namespace ap::drr

namespace ap::axpr {

template <typename ValueT, typename NodeT>
struct MethodClassImpl<ValueT, drr::tResPtn<drr::OpPatternCtx<ValueT, NodeT>>>
    : public drr::ResPtnOpPatternCtxMethodClass<ValueT, NodeT> {};

template <typename ValueT, typename NodeT>
struct MethodClassImpl<
    ValueT,
    TypeImpl<drr::tResPtn<drr::OpPatternCtx<ValueT, NodeT>>>> {};

}  // namespace ap::axpr
