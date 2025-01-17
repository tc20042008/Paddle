// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/callable_helper.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/paddle/pass/ap_drr_helper.h"
#include "paddle/ap/include/paddle/pass/ap_lower_fusion_op_pass.h"
#include "paddle/ap/include/paddle/pass/ir_helper.h"
#include "paddle/ap/include/paddle/pir/op_dialect.h"
#include "paddle/ap/include/paddle/pir/packed_ir_op_inner_source_pattern_helper.h"
#include "paddle/ap/include/paddle/pir/pass_manager_method_class.h"
#include "paddle/ap/include/paddle/pir/pass_method_class.h"
#include "paddle/ap/include/paddle/pir/program_method_class.h"
#include "paddle/ap/include/paddle/pir_node_helper.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/general/dead_code_elimination_pass.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

namespace ap::paddle {

struct PirHelperMethodClass {
  using This = PirHelperMethodClass;
  using Self = ap::paddle::IrHelper;

  static adt::Result<axpr::Value> ToString(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const void* ptr = self.shared_ptr().get();
    std::ostringstream ss;
    ss << "<PirHelper object at " << ptr << ">";
    return ss.str();
  }

  static adt::Result<axpr::Value> CreatePassManager(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    auto* ctx = ::pir::IrContext::Instance();
    ctx->GetOrRegisterDialect<ap::dialect::OperatorDialect>();
    PassManager pass_manager{std::make_shared<::pir::PassManager>(ctx, 3)};
    return GetPirPassManagerClass().New(pass_manager);
  }

  static adt::Result<axpr::Value> CreateAccessTopoDrrPass(
      axpr::InterpreterBase<axpr::Value>* interpreter,
      const axpr::Value& self_val,
      const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "create_ap_drr_pass() takes 1 arguments, but " +
        std::to_string(args.size()) + " were given"};
    std::optional<std::unique_ptr<pir::Pass>> opt_pass;
    if (args.at(0).template CastableTo<std::string>()) {
      ADT_LET_CONST_REF(drr_pass_tag_name,
                        args.at(0).template CastTo<std::string>());
      opt_pass = cinn::dialect::ir::CreateAccessTopoDrrPass(
          interpreter->circlable_ref_list(),
          drr_pass_tag_name,
          /*steps_limit=*/std::nullopt);
    } else {
      opt_pass = cinn::dialect::ir::CreateCustomAccessTopoDrrPass(
          interpreter->circlable_ref_list(),
          args.at(0),
          /*steps_limit=*/std::nullopt);
    }
    if (!opt_pass.has_value()) {
      return adt::Nothing{};
    }
    Pass pass{std::move(opt_pass.value())};
    return GetPirPassClass().New(pass);
  }

  static adt::Result<axpr::Value> CreateAccessTopoDrrOneStepPass(
      axpr::InterpreterBase<axpr::Value>* interpreter,
      const axpr::Value& self_val,
      const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "create_ap_drr_pass() takes 1 arguments, but " +
        std::to_string(args.size()) + " were given"};
    std::optional<std::unique_ptr<pir::Pass>> opt_pass;
    if (args.at(0).template CastableTo<std::string>()) {
      ADT_LET_CONST_REF(drr_pass_tag_name,
                        args.at(0).template CastTo<std::string>());
      opt_pass = cinn::dialect::ir::CreateAccessTopoDrrPass(
          interpreter->circlable_ref_list(),
          drr_pass_tag_name,
          /*steps_limit=*/1);
    } else {
      opt_pass = cinn::dialect::ir::CreateCustomAccessTopoDrrPass(
          interpreter->circlable_ref_list(),
          args.at(0),
          /*steps_limit=*/1);
    }
    if (!opt_pass.has_value()) {
      return adt::Nothing{};
    }
    Pass pass{std::move(opt_pass.value())};
    return GetPirPassClass().New(pass);
  }

  static adt::Result<axpr::Value> CreateDeadCodeEliminationPass(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0) << adt::errors::TypeError{
        std::string() + "create_dce_pass() takes 0 arguments, but " +
        std::to_string(args.size()) + " were given"};
    Pass pass{pir::CreateDeadCodeEliminationPass()};
    return GetPirPassClass().New(pass);
  }

  static adt::Result<axpr::Value> CopyFusedOpsToProgram(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "copy_fused_ops_to_program() takes 1 arguments, but " +
        std::to_string(args.size()) + " were given"};
    ADT_LET_CONST_REF(pir_node, PirNodeHelper{}.CastFromAxprValue(args.at(0)))
        << adt::errors::TypeError{
               std::string() +
               "the first argument of copy_fused_ops_to_program() must be a "
               "PackedIrOp/OptPackedIrOp (not " +
               axpr::GetTypeName(args.at(0)) + ")"};
    using RetT = adt::Result<axpr::Value>;
    return pir_node.Match(
        [&](const PackedIrOp& packed_ir_op) -> RetT {
          return This{}.CopyPackedIrOpBlockToProgram(packed_ir_op);
        },
        [&](const auto&) -> RetT {
          return adt::errors::TypeError{
              std::string() +
              "the first argument of copy_fused_ops_to_program() must be a "
              "PackedIrOp (not " +
              axpr::GetTypeName(args.at(0)) + ")"};
        });
  }

  adt::Result<axpr::Value> CopyPackedIrOpBlockToProgram(
      const PackedIrOp& packed_ir_op) {
    auto* block = packed_ir_op.fusion_op.block();
    pir::IrContext* ctx = ::pir::IrContext::Instance();
    auto new_program = std::make_shared<::pir::Program>(ctx);
    auto clone_options = ::pir::CloneOptions::All();
    pir::IrMapping ir_mapping;
    ADT_RETURN_IF_ERR(InitIrMapping(
        pir::GetUsedExternalValue(*block), &ir_mapping, new_program->block()));
    for (const auto& op : *block) {
      auto* new_op = op.Clone(ir_mapping, clone_options);
      new_program->block()->push_back(new_op);
    }
    Program ap_program{new_program};
    return GetPirProgramClass().New(ap_program);
  }

  adt::Result<adt::Ok> InitIrMapping(const std::vector<pir::Value>& free_values,
                                     pir::IrMapping* ir_mapping,
                                     pir::Block* block) {
    int i = 0;
    pir::Builder builder(pir::IrContext::Instance(), block);
    for (const auto& free_value : free_values) {
      std::string name = std::string("in") + std::to_string(i++);
      ADT_CHECK(free_value.type().isa<pir::DenseTensorType>());
      const auto& type = free_value.type().dyn_cast<pir::DenseTensorType>();
      const auto& dims = ::common::vectorize(type.dims());
      auto phi_type = ::paddle::dialect::TransToPhiDataType(type.dtype());
      auto op = builder.Build<::paddle::dialect::DataOp>(
          name, dims, phi_type, phi::Place());
      ir_mapping->Add(free_value, op->result(0));
    }
    return adt::Ok{};
  }

  static adt::Result<axpr::Value> Match(
      axpr::InterpreterBase<axpr::Value>* interpreter,
      const axpr::Value& self_val,
      const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 2) << adt::errors::TypeError{
        std::string() + "PirHelper.match() takes 2 arguments, but " +
        std::to_string(args.size()) + " were given"};
    ADT_LET_CONST_REF(program, args.at(0).template CastTo<Program>())
        << adt::errors::TypeError{std::string() +
                                  "the argument 1 of PirHelper.match() should "
                                  "b a PirProgram (not " +
                                  axpr::GetTypeName(args.at(0)) + ")"};
    ADT_CHECK(axpr::CallableHelper{}.IsCallable(args.at(1)))
        << adt::errors::TypeError{std::string() +
                                  "the argument 2 of PirHelper.match() should "
                                  "be callable object (not " +
                                  axpr::GetTypeName(args.at(1)) + ")"};
    std::vector<axpr::Value> src_ptn_func_args{std::string("fake_pass"),
                                               args.at(1)};
    ADT_LET_CONST_REF(lambda, This{}.GetDrrCtxMaker());
    axpr::Function<axpr::SerializableValue> function{lambda, std::nullopt};
    ADT_LET_CONST_REF(
        drr_ctx,
        cinn::dialect::ir::ApDrrHelper{interpreter->circlable_ref_list()}
            .InterpretDrrCtxMaker(function, src_ptn_func_args));
    ADT_CHECK(drr_ctx->source_pattern_ctx.has_value());
    ap::paddle::PackedIrOpInnerSourcePatternHelper src_pattern_helper{};
    ADT_LET_CONST_REF(
        opt_graph_match_ctx,
        src_pattern_helper.Match(program->pir_program->block(),
                                 drr_ctx->source_pattern_ctx.value()));
    return opt_graph_match_ctx.has_value();
  }

  adt::Result<axpr::Lambda<axpr::CoreExpr>> GetDrrCtxMaker() {
    using LambdaT = adt::Result<axpr::Lambda<axpr::CoreExpr>>;
    static LambdaT lambda([]() -> LambdaT {
      auto GetBody = [&](auto& ctx) -> axpr::AnfExpr {
        auto& drr_ctx = ctx.Var("DrrCtx").Call();
        drr_ctx.Attr("init_pass_name").Call(ctx.Var("pass_name"));
        drr_ctx.Attr("init_source_pattern").Call(ctx.Var("src_ptn_func"));
        return drr_ctx;
      };
      axpr::LambdaExprBuilder lmbd;
      const auto& anf_expr =
          lmbd.Lambda({"pass_name", "src_ptn_func"}, GetBody);
      const auto& core_expr = axpr::ConvertAnfExprToCoreExpr(anf_expr);
      ADT_LET_CONST_REF(
          atomic, core_expr.template TryGet<axpr::Atomic<axpr::CoreExpr>>());
      return atomic.template TryGet<axpr::Lambda<axpr::CoreExpr>>();
    }());
    return lambda;
  }
};

inline axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetPirHelperClass() {
  using Impl = PirHelperMethodClass;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>("PirHelper", [&](const auto& Yield) {
        Yield("__str__", &Impl::ToString);
        Yield("create_pass_manager", &Impl::CreatePassManager);
        Yield("create_access_topo_drr_pass", &Impl::CreateAccessTopoDrrPass);
        Yield("create_access_topo_drr_one_step_pass",
              &Impl::CreateAccessTopoDrrOneStepPass);
        Yield("create_dce_pass", &Impl::CreateDeadCodeEliminationPass);
        Yield("copy_fused_ops_to_program", &Impl::CopyFusedOpsToProgram);
        Yield("match", &Impl::Match);
      }));
  return axpr::MakeGlobalNaiveClassOps<ap::paddle::IrHelper>(cls);
}

}  // namespace ap::paddle
