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

#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/ap_lower_fusion_op_pass.h"

#include "ap/axpr/anf_expr_util.h"
#include "ap/axpr/atomic.h"
#include "ap/axpr/lambda_expr_builder.h"
#include "ap/drr/drr_graph_descriptor.h"
#include "ap/drr/drr_node_descriptor.h"
#include "ap/drr/drr_value.h"
#include "ap/drr/res_ptn_packed_ir_op_declare_data.h"
#include "ap/graph/graph_helper.h"
#include "ap/index_expr/valid_index_expr_builder.h"
#include "ap/ir_match/graph_matcher.h"
#include "ap/ir_match/ir_match_ctx.h"
#include "ap/kernel_define/compiletime_value.h"
#include "ap/paddle/indexed_ir_graph_util.h"
#include "ap/paddle/pir_graph_descriptor.h"
#include "ap/paddle/pir_node.h"
#include "ap/paddle/pir_node_descriptor.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/ap_drr_helper.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/ap_kernel_define_helper.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/ap_registry_helper.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace cinn::dialect::ir {

namespace {

using ap::paddle::PirNode;

using DrrValue = ap::drr::Value;
using DrrNode = ap::drr::Node<DrrValue>;

using DrrCtx = ap::drr::DrrCtx<DrrValue, DrrNode>;

using DrrNativeIrValue = ap::drr::NativeIrValue<DrrNode>;
using DrrPackedIrValue = ap::drr::PackedIrValue<DrrNode>;
using DrrIrValue = ap::drr::IrValue<DrrNode>;

using DrrNativeIrOp = ap::drr::NativeIrOp<DrrValue, DrrNode>;
using DrrPackedIrOp = ap::drr::PackedIrOp<DrrValue, DrrNode>;

using DrrIrOpImpl = std::variant<DrrNativeIrOp, DrrPackedIrOp>;

using IrMatchCtx = ap::ir_match::IrMatchCtx<PirNode>;

using ap::axpr::AnfExpr;
using ap::kernel_define::Module;

struct DrrIrOp : public DrrIrOpImpl {
  using DrrIrOpImpl::DrrIrOpImpl;
  DEFINE_ADT_VARIANT_METHODS(DrrIrOpImpl);
};
using DrrGraphNode = ap::graph::Node<DrrNode>;
using GraphMatchCtx = ap::ir_match::GraphMatchCtx<PirNode, DrrGraphNode>;

adt::Result<DrrNode> GetApDrrAnchor(const DrrCtx& drr_ctx) {
  ADT_LET_CONST_REF(src_ptn_ctx, drr_ctx->GetSourcePatternCtx());
  auto ptn_node_area = src_ptn_ctx->node_arena;
  ap::graph::GraphDescriptor<DrrGraphNode> source_pattern_graph{};
  ADT_CHECK(ptn_node_area->nodes().size() > 0);
  ap::graph::GraphHelper<DrrGraphNode> graph_helper(source_pattern_graph);
  const auto& start_ptn_node = ptn_node_area->nodes().at(0).node();
  ADT_LET_CONST_REF(anchor_node, graph_helper.FindAnchor(start_ptn_node));
  ADT_LET_CONST_REF(anchor, anchor_node.Get());
  return anchor;
}

std::optional<DrrIrValue> CastToDrrIrValue(const DrrNode& drr_node) {
  return drr_node.Match(
      [](const DrrNativeIrValue& ir_value) -> std::optional<DrrIrValue> {
        return DrrIrValue{ir_value};
      },
      [](const DrrPackedIrValue& ir_value) -> std::optional<DrrIrValue> {
        return DrrIrValue{ir_value};
      },
      [](const auto&) -> std::optional<DrrIrValue> { return std::nullopt; });
}

adt::Result<std::vector<DrrIrValue>> GetResPtnOutputs(const DrrCtx& drr_ctx) {
  std::vector<DrrIrValue> ret;
  ADT_LET_CONST_REF(res_ptn_ctx, drr_ctx->GetResultPatternCtx());
  const auto& nodes = res_ptn_ctx->node_arena->nodes();
  for (const auto& drr_node : nodes) {
    ADT_LET_CONST_REF(downstreams, drr_node.node().DownstreamNodes());
    if (downstreams.size() == 0) {
      const auto& opt_drr_ir_value = CastToDrrIrValue(drr_node);
      ADT_CHECK(opt_drr_ir_value.has_value());
      ret.push_back(opt_drr_ir_value.value());
    }
  }
  return ret;
}

struct ApLowerFusionOpPatternCtx {
  DrrCtx drr_ctx;
  std::vector<DrrIrValue> res_ptn_outputs;
  DrrNode anchor;
  std::string anchor_op_name;

  static adt::Result<ApLowerFusionOpPatternCtx> MakeFromDrrCtx(
      const DrrCtx& drr_ctx) {
    ADT_LET_CONST_REF(res_ptn_outputs, GetResPtnOutputs(drr_ctx));
    ADT_LET_CONST_REF(anchor, GetApDrrAnchor(drr_ctx));
    ADT_LET_CONST_REF(
        anchor_op_name,
        anchor.Match(
            [&](const DrrNativeIrOp& ir_op) -> adt::Result<std::string> {
              return ir_op->op_declare->op_name;
            },
            [&](const DrrPackedIrOp& ir_op) -> adt::Result<std::string> {
              return PirNode::GetOpNameFromDrrPackedOpName(
                  ir_op->op_declare->op_name);
            },
            [&](const auto&) -> adt::Result<std::string> {
              return adt::errors::TypeError{
                  "anchor drr node should be a op node but value node found."};
            }));
    return ApLowerFusionOpPatternCtx{
        drr_ctx, res_ptn_outputs, anchor, anchor_op_name};
  }
};

class ApLowerFusionOpPattern : public pir::RewritePattern {
 private:
  ApLowerFusionOpPatternCtx ctx_;

 public:
  ApLowerFusionOpPattern(pir::IrContext* ir_context,
                         const ApLowerFusionOpPatternCtx& ctx)
      : pir::RewritePattern(ctx.anchor_op_name, 1, ir_context, {}), ctx_(ctx) {}

  bool MatchAndRewrite(
      pir::Operation* op,
      pir::PatternRewriter& rewriter) const override {  // // NOLINT
    const auto& ret = TryMatchAndRewrite(op, &rewriter);
    if (ret.HasError()) {
      LOG(ERROR) << "\nTraceback (most recent call last):\n"
                 << ret.GetError().CallStackToString() << "\n"
                 << ret.GetError().class_name() << ": " << ret.GetError().msg();
      return false;
    }
    return ret.GetOkValue();
  }

  adt::Result<bool> TryMatchAndRewrite(pir::Operation* op,
                                       pir::PatternRewriter* rewriter) const {
    ADT_LET_CONST_REF(match_ctx, GetMatchCtx(op));
    ADT_CHECK(ctx_.drr_ctx->pass_name.has_value());
    LOG(ERROR) << "drr: " << ctx_.drr_ctx->pass_name.value() << " matched.";
    return RewriteByResultPattern(match_ctx, op->GetParent(), rewriter);
  }

  adt::Result<GraphMatchCtx> GetMatchCtx(pir::Operation* op) const {
    auto* parent_block = op->GetParent();
    ADT_CHECK(parent_block != nullptr);
    auto* parent_op = parent_block->GetParentOp();
    ADT_CHECK(!parent_op->isa<cinn::dialect::FusionOp>());
    const auto& anchor = ctx_.anchor;
    ap::graph::GraphDescriptor<PirNode> pir_graph{};
    ap::graph::GraphDescriptor<DrrGraphNode> src_ptn_graph{};
    ap::ir_match::GraphMatcher<PirNode, DrrGraphNode> graph_matcher(
        pir_graph, src_ptn_graph);
    ADT_LET_CONST_REF(anchor_cstr,
                      src_ptn_graph.GetNodeConstraint(anchor.node()));
    const auto& obj_node = CastToPirNode(op);
    ADT_LET_CONST_REF(satisfy_constraint,
                      pir_graph.Satisfy(obj_node, anchor_cstr));
    ADT_CHECK(satisfy_constraint) << adt::errors::ValueError{
        "pir_graph.Satisfy(obj_node, anchor_cstr) test failed."};
    ADT_LET_CONST_REF(graph_ctx,
                      graph_matcher.MatchByAnchor(obj_node, anchor.node()));
    ADT_LET_CONST_REF(graph_matched,
                      graph_matcher.IsGraphMatched(graph_ctx, anchor.node()));
    ADT_CHECK(graph_matched);
    return graph_ctx;
  }

  PirNode CastToPirNode(pir::Operation* op) const {
    if (op->isa<cinn::dialect::FusionOp>()) {
      ap::paddle::PackedIrOp ir_op{op->dyn_cast<cinn::dialect::FusionOp>()};
      return ir_op;
    } else {
      ap::paddle::NativeIrOp ir_op{op};
      return ir_op;
    }
  }

  adt::Result<bool> RewriteByResultPattern(
      const GraphMatchCtx& match_ctx,
      pir::Block* block,
      pir::PatternRewriter* rewriter) const {
    std::set<pir::Operation*> new_ops;
    ADT_LET_CONST_REF(rewrited,
                      TryRewriteByResultPattern(match_ctx, &new_ops, rewriter));
    return rewrited;
  }

  struct RewriteCtx {
    const std::unordered_map<pir::Operation*, std::size_t>
        matched_op2order_value;
    std::unordered_map<std::string, pir::Value> name2native_value;
    std::unordered_map<std::string, std::vector<pir::Value>> name2packed_value;

    adt::Result<std::size_t> GetMatchedOpOrderValue(pir::Operation* op) const {
      const auto iter = this->matched_op2order_value.find(op);
      if (iter == this->matched_op2order_value.end()) {
        return adt::errors::IndexError{
            "RewriteCtx::GetMatchedOpOrderValue failed."};
      }
      return iter->second;
    }

    adt::Result<pir::Value> GetNativeValue(
        const std::string& value_name) const {
      const auto iter = this->name2native_value.find(value_name);
      if (iter == this->name2native_value.end()) {
        return adt::errors::IndexError{"RewriteCtx::GetNativeValue failed."};
      }
      return iter->second;
    }
  };

  adt::Result<std::unordered_set<pir::Operation*>> GetMatchedOps(
      const GraphMatchCtx& match_ctx) const {
    ADT_LET_CONST_REF(src_ptn_ctx, ctx_.drr_ctx->GetSourcePatternCtx());
    const auto& nodes = src_ptn_ctx->node_arena->nodes();
    std::unordered_set<pir::Operation*> ops;
    for (const auto& drr_node : nodes) {
      ADT_LET_CONST_REF(pir_node,
                        match_ctx->GetSoleBigGraphNode(drr_node.node()));
      const auto& opt_op = CastToPirOp(pir_node);
      if (opt_op.has_value()) {
        ADT_CHECK(ops.emplace(opt_op.value()).second);
      }
    }
    return ops;
  }

  std::optional<pir::Operation*> CastToPirOp(const PirNode& pir_node) const {
    return pir_node.Match(
        [](const ap::paddle::NativeIrOp& ir_op)
            -> std::optional<pir::Operation*> { return ir_op.op; },
        [&](const ap::paddle::PackedIrOp& ir_op)
            -> std::optional<pir::Operation*> {
          return static_cast<pir::Operation*>(ir_op.fusion_op);
        },
        [&](const auto&) -> std::optional<pir::Operation*> {
          return std::nullopt;
        });
  }

  adt::Result<std::unordered_map<pir::Operation*, std::size_t>>
  MakeMatchedOp2OrderValue(const GraphMatchCtx& match_ctx) const {
    ADT_LET_CONST_REF(ops, GetMatchedOps(match_ctx));
    std::unordered_map<pir::Operation*, std::size_t> ret;
    pir::Operation* start = *ops.begin();
    auto* block = start->GetParent();
    pir::Block::Iterator left_iter = *start;
    pir::Block::Iterator right_iter = *start;
    for (int i = 0; ret.size() < ops.size() && i < block->size(); ++i) {
      if (ops.count(&*left_iter) > 0) {
        ret[&*left_iter] = -i;
      }
      if (ops.count(&*right_iter) > 0) {
        ret[&*right_iter] = i;
      }
      if (&*left_iter != &block->front()) {
        --left_iter;
      }
      if (&*right_iter != &block->back()) {
        ++right_iter;
      }
    }
    ADT_CHECK(ret.size() == ops.size());
    return ret;
  }

  adt::Result<bool> TryRewriteByResultPattern(
      const GraphMatchCtx& match_ctx,
      std::set<pir::Operation*>* new_ops,
      pir::PatternRewriter* rewriter) const {
    ADT_LET_CONST_REF(matched_op2order_value,
                      MakeMatchedOp2OrderValue(match_ctx));
    RewriteCtx rewrite_ctx{matched_op2order_value, {}, {}};
    auto Build = [&](const auto& res_ptn_op) -> adt::Result<adt::Ok> {
      return BuildNewOp(rewriter, new_ops, res_ptn_op, &rewrite_ctx, match_ctx);
    };
    ADT_RETURN_IF_ERR(VisitEachResPtnOp(Build));
    ADT_RETURN_IF_ERR(
        ReplaceOutputResPtnTensor(match_ctx, rewrite_ctx, rewriter));
    return true;
  }

  adt::Result<adt::Ok> ReplaceOutputResPtnTensor(
      const GraphMatchCtx& match_ctx,
      const RewriteCtx& rewrite_ctx,
      pir::PatternRewriter* rewriter) const {
    auto Replace = [&](pir::Value from, pir::Value to) -> adt::Result<adt::Ok> {
      rewriter->ReplaceAllUsesWith(from, to);
      return adt::Ok{};
    };
    return VisitOutputPirValueReplacementPair(match_ctx, rewrite_ctx, Replace);
  }

  template <typename DoEachPairT>
  adt::Result<adt::Ok> VisitOutputPirValueReplacementPair(
      const GraphMatchCtx& match_ctx,
      const RewriteCtx& rewrite_ctx,
      const DoEachPairT& DoEachPair) const {
    for (const auto& res_ptn_drr_ir_value : ctx_.res_ptn_outputs) {
      const auto& opt_drr_ir_value =
          SrcPtnIrValue4ResPtnIrValue(res_ptn_drr_ir_value);
      ADT_CHECK(opt_drr_ir_value.has_value());
      const auto& drr_ir_value = opt_drr_ir_value.value();
      const auto& ret = drr_ir_value.Match(
          [&](const DrrNativeIrValue& native_ir_value) -> adt::Result<adt::Ok> {
            ADT_LET_CONST_REF(
                pir_node,
                match_ctx->GetSoleBigGraphNode(native_ir_value->node));
            ADT_LET_CONST_REF(
                pir_value,
                pir_node.template TryGet<ap::paddle::NativeIrValue>());
            pir::Value from = pir_value.value;
            ADT_LET_CONST_REF(
                to, rewrite_ctx.GetNativeValue(native_ir_value->name));
            return DoEachPair(from, to);
          },
          [&](const DrrPackedIrValue& ir_value) -> adt::Result<adt::Ok> {
            return adt::errors::NotImplementedError{
                "PackedIrValue replacement is not supoorted yet."};
          });
      ADT_RETURN_IF_ERR(ret);
    }
    return adt::Ok{};
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitEachResPtnOp(const DoEachT& DoEach) const {
    auto DoEachResPtnOp =
        [&](const auto& res_ptn_node) -> adt::Result<adt::Ok> {
      const auto& opt_res_ptn_op = ConvertToResPtnOp(res_ptn_node);
      if (opt_res_ptn_op.has_value()) {
        ADT_RETURN_IF_ERR(DoEach(opt_res_ptn_op.value()));
      }
      return adt::Ok{};
    };
    return VisitEachResPtnNode(DoEachResPtnOp);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitEachResPtnNode(const DoEachT& DoEach) const {
    ADT_LET_CONST_REF(res_ptn_ctx, ctx_.drr_ctx->GetResultPatternCtx());
    for (const auto& drr_node : res_ptn_ctx->node_arena->nodes()) {
      ADT_RETURN_IF_ERR(DoEach(drr_node));
    }
    return adt::Ok{};
  }

  std::optional<DrrIrOp> ConvertToResPtnOp(const DrrNode& drr_node) const {
    return drr_node.Match(
        [&](const DrrNativeIrOp& ir_op) -> std::optional<DrrIrOp> {
          return DrrIrOp{ir_op};
        },
        [&](const DrrPackedIrOp& ir_op) -> std::optional<DrrIrOp> {
          return DrrIrOp{ir_op};
        },
        [&](const auto&) -> std::optional<DrrIrOp> { return std::nullopt; });
  }

  adt::Result<adt::Ok> BuildNewOp(pir::PatternRewriter* rewriter,
                                  std::set<pir::Operation*>* new_ops,
                                  const DrrIrOp& res_ptn_op,
                                  RewriteCtx* rewrite_ctx,
                                  const GraphMatchCtx& match_ctx) const {
    return res_ptn_op.Match(
        [&](const DrrNativeIrOp& ir_op) -> adt::Result<adt::Ok> {
          return adt::errors::NotImplementedError{
              "building native ir op is not supported now."};
        },
        [&](const DrrPackedIrOp& ir_op) -> adt::Result<adt::Ok> {
          return BuildPackedOp(
              rewriter, new_ops, ir_op, rewrite_ctx, match_ctx);
        });
  }

  adt::Result<adt::Ok> BuildPackedOp(pir::PatternRewriter* rewriter,
                                     std::set<pir::Operation*>* new_ops,
                                     const DrrPackedIrOp& res_ptn_ir_op,
                                     RewriteCtx* rewrite_ctx,
                                     const GraphMatchCtx& match_ctx) const {
    ADT_CHECK(res_ptn_ir_op->op_declare->op_name, "ap_pattern_fusion_op");
    ADT_RETURN_IF_ERR(
        InsertInputPirValueToReplaceCtx(res_ptn_ir_op, rewrite_ctx, match_ctx));
    ADT_LET_CONST_REF(input_values,
                      GetPackedOpInputValues(res_ptn_ir_op, *rewrite_ctx));
    ADT_RETURN_IF_ERR(
        TrySetInsertPointer(rewriter, *rewrite_ctx, res_ptn_ir_op, match_ctx));
    ADT_LET_CONST_REF(combined_value,
                      InsertCombinedOp(new_ops, rewriter, input_values));
    ADT_LET_CONST_REF(kernel_define_lambda_str,
                      GetKernelDefineLambdaStr(res_ptn_ir_op, match_ctx));
    ADT_LET_CONST_REF(define_ctx_lambda_str,
                      GetDefineCtxLambdaStr(res_ptn_ir_op));
    ADT_LET_CONST_REF(kernel_dispatch_lambda_str,
                      GetKernelDispatchLambdaStr(res_ptn_ir_op));
    ADT_LET_CONST_REF(dispatch_ctx_lambda_str,
                      GetDispatchCtxLambdaStr(res_ptn_ir_op, match_ctx));
    ADT_LET_CONST_REF(num_outputs,
                      GetApKernelNumOutputs(res_ptn_ir_op, match_ctx));
    ADT_LET_CONST_REF(ap_pattern_fusion_combined_out,
                      MakeApPatternFusionOp(rewriter,
                                            new_ops,
                                            combined_value,
                                            num_outputs,
                                            kernel_define_lambda_str,
                                            define_ctx_lambda_str,
                                            kernel_dispatch_lambda_str,
                                            dispatch_ctx_lambda_str));
    ADT_LET_CONST_REF(output_values,
                      GetPackedOpOutputValues(
                          rewriter, new_ops, ap_pattern_fusion_combined_out));
    ADT_RETURN_IF_ERR(UpdateApKernelOutputsInReplaceCtx(
        output_values, res_ptn_ir_op, rewrite_ctx));
    return adt::Ok{};
  }

  adt::Result<std::string> GetDefineCtxLambdaStr(
      const DrrPackedIrOp& res_ptn_ir_op) const {
    ap::axpr::LambdaExprBuilder lmbd;
    auto ConstructLambdaBody = [&](ap::axpr::LetContext& ctx) {
      return ctx.Var("ctx");
    };
    ap::axpr::AnfExpr anf_expr = lmbd.Lambda({"ctx"}, ConstructLambdaBody);
    return anf_expr.DumpToJsonString();
  }

  adt::Result<std::string> GetDispatchCtxLambdaStr(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    ap::axpr::LambdaExprBuilder lmbd;
    auto ConstructLambdaBody = [&](auto& ctx) -> ap::axpr::AnfExpr {
      return ctx.Var("ctx").Attr("DispatcherCtx").Call(ctx.None());
    };
    ap::axpr::AnfExpr anf_expr = lmbd.Lambda({"ctx"}, ConstructLambdaBody);
    return anf_expr.DumpToJsonString();
  }

  adt::Result<std::string> GetKernelDefineLambdaStr(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    const auto& op_declare = res_ptn_ir_op->op_declare;
    ADT_LET_CONST_REF(
        data, op_declare->cast_data<ap::drr::ResPtnPackedIrOpDeclareData>());
    const auto& lambda = data->kernel_define();
    ADT_LET_CONST_REF(ap_kernel_module, GetApKernelModule(lambda, match_ctx));
    ap::axpr::AnfExpr anf_expr =
        ConvertApKernelModuleToAnfExpr(ap_kernel_module);
    return anf_expr.DumpToJsonString();
  }

  adt::Result<Module> GetApKernelModule(
      const ap::axpr::Lambda<ap::axpr::CoreExpr>& lambda,
      const GraphMatchCtx& match_ctx) const {
    ADT_LET_CONST_REF(src_ptn_ctx, ctx_.drr_ctx->GetSourcePatternCtx());
    IrMatchCtx ir_match_ctx{src_ptn_ctx, match_ctx};
    std::vector<ap::kernel_define::NamedKernelArg> named_kernel_args;
    ap::kernel_define::DefineCtx<PirNode> define_ctx{ir_match_ctx,
                                                     named_kernel_args};
    ApKernelDefineHelper helper{};
    ADT_LET_CONST_REF(m, helper.Interpret(lambda, define_ctx));
    return m;
  }

  AnfExpr ConvertApKernelModuleToAnfExpr(const Module& m) const {
    auto ConvertArgType = [&](auto& ctx, const auto& arg_type) -> AnfExpr {
      return arg_type.Match(
          [&](const ap::axpr::DataType& data_type) -> AnfExpr {
            const auto& var = ctx.Var("DataType").Attr(data_type.Name());
            return ap::axpr::tVar<std::string>{var.name()};
          },
          [&](const ap::axpr::PointerType& pointer_type) -> AnfExpr {
            const auto& var = ctx.Var("PointerType").Attr(pointer_type.Name());
            return ap::axpr::tVar<std::string>{var.name()};
          });
    };
    auto ConvertFuncDeclareCall = [&](auto& ctx,
                                      const auto& func_declare) -> AnfExpr {
      const auto& func_name = ctx.String(func_declare->func_id);
      std::vector<AnfExpr> elts;
      elts.reserve(func_declare->arg_types->size());
      for (const auto& arg_type : *func_declare->arg_types) {
        elts.emplace_back(ConvertArgType(ctx, arg_type));
      }
      const auto& arg_type_anf_expr = ctx.Call(ap::axpr::kBuiltinList(), elts);
      return ctx.Call("FuncDeclare", func_name, arg_type_anf_expr);
    };
    auto ConvertFuncDeclareList = [&](auto& ctx) -> AnfExpr {
      std::vector<AnfExpr> elts;
      elts.reserve(m->func_declares->size());
      for (const auto& func_declare : *m->func_declares) {
        elts.emplace_back(ConvertFuncDeclareCall(ctx, func_declare));
      }
      return ctx.Call(ap::axpr::kBuiltinList(), elts);
    };
    auto ConvertSourceCodeCall = [&](auto& ctx) -> AnfExpr {
      const auto& str = ctx.String(m->source_code->source_code);
      return ctx.Call("SourceCode", str);
    };
    auto ConstructLambdaBody = [&](auto& ctx) -> ap::axpr::AnfExpr {
      const auto& declare = ConvertFuncDeclareList(ctx);
      const auto& source_code = ConvertSourceCodeCall(ctx);
      return ctx.Call("Module", declare, source_code);
    };
    return ap::axpr::LambdaExprBuilder{}.Lambda({}, ConstructLambdaBody);
  }

  adt::Result<std::string> GetKernelDispatchLambdaStr(
      const DrrPackedIrOp& res_ptn_ir_op) const {
    const auto& op_declare = res_ptn_ir_op->op_declare;
    ADT_LET_CONST_REF(
        data, op_declare->cast_data<ap::drr::ResPtnPackedIrOpDeclareData>());
    const auto& lambda = data->kernel_dispatch();
    ap::axpr::AnfExpr anf_expr = ap::axpr::ConvertCoreExprToAnfExpr(lambda);
    return anf_expr.DumpToJsonString();
  }

  adt::Result<pir::Value> MakeApPatternFusionOp(
      pir::PatternRewriter* rewriter,
      std::set<pir::Operation*>* new_ops,
      pir::Value input,
      std::size_t num_outputs,
      const std::string& kernel_define_lambda_str,
      const std::string& define_ctx_lambda_str,
      const std::string& kernel_dispatch_lambda_str,
      const std::string& dispatch_ctx_lambda_str) const {
    auto ap_unary =
        rewriter->Build<paddle::dialect::ApUnaryOp>(input,
                                                    num_outputs,
                                                    kernel_define_lambda_str,
                                                    define_ctx_lambda_str,
                                                    kernel_dispatch_lambda_str,
                                                    dispatch_ctx_lambda_str);
    ADT_CHECK(new_ops->emplace(ap_unary).second);
    return ap_unary.out();
  }

  adt::Result<std::vector<pir::Value>> GetPackedOpOutputValues(
      pir::PatternRewriter* rewriter,
      std::set<pir::Operation*>* new_ops,
      pir::Value combined_out) const {
    auto split_op = rewriter->Build<pir::SplitOp>(combined_out);
    ADT_CHECK(new_ops->emplace(split_op).second);
    return split_op.outputs();
  }

  adt::Result<adt::Ok> UpdateApKernelOutputsInReplaceCtx(
      const std::vector<pir::Value>& output_values,
      const DrrPackedIrOp& res_ptn_ir_op,
      RewriteCtx* rewrite_ctx) const {
    auto UpdateRewriteCtx = [&](const DrrIrValue& ir_value,
                                const std::vector<pir::Value>& output_slice)
        -> adt::Result<adt::Ok> {
      return ir_value.Match(
          [&](const DrrNativeIrValue& ir_value) -> adt::Result<adt::Ok> {
            ADT_CHECK(output_slice.size() == 1);
            const auto& k = ir_value->name;
            const auto& v = output_slice.at(0);
            ADT_CHECK(rewrite_ctx->name2native_value.emplace(k, v).second);
            return adt::Ok{};
          },
          [&](const DrrPackedIrValue& ir_value) -> adt::Result<adt::Ok> {
            const auto& k = ir_value->name;
            const auto& v = output_slice;
            ADT_CHECK(rewrite_ctx->name2packed_value.emplace(k, v).second);
            return adt::Ok{};
          });
    };
    ADT_RETURN_IF_ERR(VisitEachMatchedDrrIrValueAndOutputSlice(
        output_values, res_ptn_ir_op, UpdateRewriteCtx));
    return adt::Ok{};
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitEachMatchedDrrIrValueAndOutputSlice(
      const std::vector<pir::Value>& output_values,
      const DrrPackedIrOp& res_ptn_ir_op,
      const DoEachT& DoEach) const {
    std::size_t offset = 0;
    auto DoEachSlice =
        [&](const DrrIrValue& drr_ir_value) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(num_ir_values, GetResPtnNumPirValues(drr_ir_value));
      ADT_CHECK(offset + num_ir_values <= output_values.size());
      std::vector<pir::Value> slice{
          output_values.begin() + offset,
          output_values.begin() + offset + num_ir_values};
      return DoEach(drr_ir_value, slice);
    };
    return VisitResPtnOutputIrValueByResPtnIrOp(res_ptn_ir_op, DoEachSlice);
  }

  adt::Result<std::size_t> GetResPtnNumPirValues(
      const DrrIrValue& drr_ir_value) const {
    return drr_ir_value.Match(
        [&](const DrrNativeIrValue&) -> adt::Result<std::size_t> { return 1; },
        [&](const DrrPackedIrValue&) -> adt::Result<std::size_t> {
          return adt::errors::NotImplementedError{
              "GetApKernelNumOutputs not support DrrPackedIrValue."};
        });
  }

  adt::Result<std::size_t> GetApKernelNumOutputs(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    std::size_t num_outputs = 0;
    auto AccNumOutputs =
        [&](const DrrIrValue& drr_ir_value) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(num_ir_values, GetResPtnNumPirValues(drr_ir_value));
      num_outputs += num_ir_values;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(
        VisitResPtnOutputIrValueByResPtnIrOp(res_ptn_ir_op, AccNumOutputs));
    return num_outputs;
  }

  adt::Result<pir::Value> InsertCombinedOp(
      std::set<pir::Operation*>* new_ops,
      pir::PatternRewriter* rewriter,
      const std::vector<pir::Value>& inputs) const {
    auto combined_op = rewriter->Build<pir::CombineOp>(inputs);
    ADT_CHECK(new_ops->emplace(combined_op).second);
    return combined_op.out();
  }

  adt::Result<adt::Ok> TrySetInsertPointer(
      pir::PatternRewriter* rewriter,
      const RewriteCtx& rewrite_ctx,
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    ADT_LET_CONST_REF(
        opt_last_pir_op,
        GetLastMatchedPirOp(rewrite_ctx, res_ptn_ir_op, match_ctx));
    if (opt_last_pir_op.has_value()) {
      rewriter->SetInsertionPointAfter(opt_last_pir_op.value());
    }
    return adt::Ok{};
  }

  adt::Result<std::optional<pir::Operation*>> GetLastMatchedPirOp(
      const RewriteCtx& rewrite_ctx,
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    std::optional<pir::Operation*> last_op;
    std::optional<std::size_t> op_order_value;
    auto UpdatePirOp = [&](pir::Operation* op) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(order_value, rewrite_ctx.GetMatchedOpOrderValue(op));
      if (!op_order_value.has_value() || op_order_value.value() < order_value) {
        op_order_value = order_value;
        last_op = op;
      }
      return adt::Ok{};
    };
    auto UpdateLastOp = [&](const DrrGraphNode& op) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(pir_node, match_ctx->GetSoleBigGraphNode(op));
      return pir_node.Match(
          [&](const ap::paddle::NativeIrOp& ir_op) -> adt::Result<adt::Ok> {
            return UpdatePirOp(ir_op.op);
          },
          [&](const ap::paddle::PackedIrOp& ir_op) -> adt::Result<adt::Ok> {
            return UpdatePirOp(ir_op.fusion_op);
          },
          [](const auto&) -> adt::Result<adt::Ok> { return adt::Ok{}; });
    };
    auto DoEachOutput = [&](const DrrIrValue& output) -> adt::Result<adt::Ok> {
      const auto& opt_src_ptn_ir_value = SrcPtnIrValue4ResPtnIrValue(output);
      if (!opt_src_ptn_ir_value.has_value()) {
        return adt::Ok{};
      }
      const auto& src_ptn_output = opt_src_ptn_ir_value.value();
      ADT_LET_CONST_REF(output_upstreams,
                        src_ptn_output.node().UpstreamNodes());
      ADT_LET_CONST_REF(op_result, output_upstreams.Sole());
      ADT_LET_CONST_REF(op_result_upstreams, op_result.UpstreamNodes());
      ADT_LET_CONST_REF(ir_op, op_result_upstreams.Sole());
      return UpdateLastOp(ir_op);
    };
    ADT_RETURN_IF_ERR(
        VisitResPtnOutputIrValueByResPtnIrOp(res_ptn_ir_op, DoEachOutput));
    return last_op;
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitResPtnInputIrValueByResPtnIrOp(
      const DrrPackedIrOp& res_ptn_ir_op, const DoEachT& DoEach) const {
    auto VisitOpOperand =
        [&](const DrrGraphNode& op_operand) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(op_operand_downstreams, op_operand.UpstreamNodes());
      ADT_LET_CONST_REF(ir_value_node, op_operand_downstreams.Sole());
      ADT_LET_CONST_REF(ir_value, ir_value_node.Get());
      const auto& opt_drr_ir_value = CastToDrrIrValue(ir_value);
      ADT_CHECK(opt_drr_ir_value.has_value());
      const auto& drr_ir_value = opt_drr_ir_value.value();
      return DoEach(drr_ir_value);
    };
    ADT_LET_CONST_REF(upstreams, res_ptn_ir_op->node.UpstreamNodes());
    ADT_RETURN_IF_ERR(upstreams.VisitNodes(VisitOpOperand));
    return adt::Ok{};
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitResPtnOutputIrValueByResPtnIrOp(
      const DrrPackedIrOp& res_ptn_ir_op, const DoEachT& DoEach) const {
    auto VisitOpResult =
        [&](const DrrGraphNode& op_result) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(op_result_downstreams, op_result.DownstreamNodes());
      ADT_LET_CONST_REF(ir_node, op_result_downstreams.Sole());
      ADT_LET_CONST_REF(drr_ir_node, ir_node.Get());
      const auto& opt_drr_ir_value = CastToDrrIrValue(drr_ir_node);
      ADT_CHECK(opt_drr_ir_value.has_value());
      const auto& drr_ir_value = opt_drr_ir_value.value();
      return DoEach(drr_ir_value);
    };
    ADT_LET_CONST_REF(downstreams, res_ptn_ir_op->node.DownstreamNodes());
    ADT_RETURN_IF_ERR(downstreams.VisitNodes(VisitOpResult));
    return adt::Ok{};
  }

  std::optional<DrrIrValue> SrcPtnIrValue4ResPtnIrValue(
      const DrrIrValue& res_ptn_ir_value) const {
    const auto& opt_src_ptn_ctx = ctx_.drr_ctx->GetSourcePatternCtx();
    if (opt_src_ptn_ctx.HasError()) {
      return std::nullopt;
    }
    const auto& src_ptn_ctx = opt_src_ptn_ctx.GetOkValue();
    const auto& map = src_ptn_ctx->tensor_pattern_ctx->uid2ir_value;
    auto GetSrcPtnIrValue =
        [&](const auto& ir_value) -> std::optional<DrrIrValue> {
      const auto iter = map.find(ir_value->name);
      if (iter == map.end()) {
        return std::nullopt;
      }
      return iter->second;
    };
    return res_ptn_ir_value.Match(
        [&](const DrrNativeIrValue& ir_value) -> std::optional<DrrIrValue> {
          return GetSrcPtnIrValue(ir_value);
        },
        [&](const DrrPackedIrValue& ir_value) -> std::optional<DrrIrValue> {
          return GetSrcPtnIrValue(ir_value);
        });
  }

  adt::Result<adt::Ok> InsertInputPirValueToReplaceCtx(
      const DrrPackedIrOp& res_ptn_ir_op,
      RewriteCtx* rewrite_ctx,
      const GraphMatchCtx& match_ctx) const {
    auto InitInput =
        [&](const DrrIrValue& drr_ir_value) -> adt::Result<adt::Ok> {
      return drr_ir_value.Match(
          [&](const DrrNativeIrValue& res_ptn_ir_value)
              -> adt::Result<adt::Ok> {
            const auto iter =
                rewrite_ctx->name2native_value.find(res_ptn_ir_value->name);
            if (iter != rewrite_ctx->name2native_value.end()) {
              return adt::Ok{};
            }
            const auto& opt_ir_value =
                SrcPtnIrValue4ResPtnIrValue(res_ptn_ir_value);
            ADT_CHECK(opt_ir_value.has_value());
            const auto& ir_value = opt_ir_value.value();
            ADT_LET_CONST_REF(pir_node,
                              match_ctx->GetSoleBigGraphNode(ir_value.node()));
            ADT_LET_CONST_REF(
                pir_value,
                pir_node.template TryGet<ap::paddle::NativeIrValue>())
                << adt::errors::TypeError{
                       "pir_node is not an ap::paddle::NativeIrValue"};
            rewrite_ctx->name2native_value[ir_value.name()] = pir_value.value;
            return adt::Ok{};
          },
          [&](const DrrPackedIrValue ir_value) -> adt::Result<adt::Ok> {
            return adt::errors::NotImplementedError{
                "packed input ir values are not supported yet."};
          });
    };
    return VisitResPtnInputIrValueByResPtnIrOp(res_ptn_ir_op, InitInput);
  }

  adt::Result<std::vector<pir::Value>> GetPackedOpInputValues(
      const DrrPackedIrOp& res_ptn_ir_op, const RewriteCtx& rewrite_ctx) const {
    std::vector<pir::Value> ret;
    auto CollectValues = [&](pir::Value value) -> adt::Result<adt::Ok> {
      ret.push_back(value);
      return adt::Ok{};
    };
    auto VisitAndCollect =
        [&](const DrrIrValue& drr_ir_value) -> adt::Result<adt::Ok> {
      return VisitPirValueByIrValue(drr_ir_value, rewrite_ctx, CollectValues);
    };
    ADT_RETURN_IF_ERR(
        VisitResPtnInputIrValueByResPtnIrOp(res_ptn_ir_op, VisitAndCollect));
    return ret;
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitPirValueByIrValue(const DrrIrValue& ir_value,
                                              const RewriteCtx& rewrite_ctx,
                                              const DoEachT& DoEach) const {
    ADT_RETURN_IF_ERR(ir_value.Match(
        [&](const DrrNativeIrValue& ir_value) -> adt::Result<adt::Ok> {
          const auto& name = ir_value->name;
          const auto& iter = rewrite_ctx.name2native_value.find(name);
          ADT_CHECK(iter != rewrite_ctx.name2native_value.end());
          return DoEach(iter->second);
        },
        [&](const DrrPackedIrValue& ir_value) -> adt::Result<adt::Ok> {
          const auto& name = ir_value->name;
          const auto& iter = rewrite_ctx.name2packed_value.find(name);
          ADT_CHECK(iter != rewrite_ctx.name2packed_value.end());
          for (const auto& value : iter->second) {
            ADT_RETURN_IF_ERR(DoEach(value));
          }
          return adt::Ok{};
        }));
    return adt::Ok{};
  }
};

class ApLowerFusionOpPass : public pir::PatternRewritePass {
 public:
  ApLowerFusionOpPass()
      : pir::PatternRewritePass("ap_lower_fusion_op_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    const auto& ret = TryInitializePatterns(&ps, context);
    if (ret.HasError()) {
      LOG(ERROR) << "\nTraceback (most recent call last):\n"
                 << ret.GetError().CallStackToString() << "\n"
                 << "InitializePatterns " << ret.GetError().class_name() << ": "
                 << ret.GetError().msg();
    }
    return ps;
  }

  adt::Result<adt::Ok> TryInitializePatterns(pir::RewritePatternSet* ps,
                                             pir::IrContext* context) {
    auto AddFusionOpPattern = [&](const auto& drr_ctx) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(pattern_ctx,
                        ApLowerFusionOpPatternCtx::MakeFromDrrCtx(drr_ctx));
      ps->Add(std::make_unique<ApLowerFusionOpPattern>(context, pattern_ctx));
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitEachDrrCtx(AddFusionOpPattern));
    return adt::Ok{};
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitEachDrrCtx(const DoEachT& DoEach) {
    ADT_LET_CONST_REF(registry, ApRegistryHelper{}.SingltonRegistry());
    const auto& drr_registry_items = registry->drr_registry_items;
    for (const auto& [drr_pass_name, nice2drr_items] : drr_registry_items) {
      std::optional<DrrCtx> opt_drr_ctx;
      for (const auto& [nice, drr_items] : nice2drr_items) {
        if (opt_drr_ctx.has_value()) {
          break;
        }
        for (const auto& drr_item : drr_items) {
          const auto& drr_ctx = GetDrrCtx(drr_pass_name, drr_item);
          if (drr_ctx.HasOkValue()) {
            ADT_RETURN_IF_ERR(DoEach(drr_ctx.GetOkValue()));
            opt_drr_ctx = drr_ctx.GetOkValue();
            break;
          } else {
            LOG(ERROR) << "\nTraceback (most recent call last):\n"
                       << drr_ctx.GetError().CallStackToString() << "\n"
                       << drr_ctx.GetError().class_name()
                       << ": drr_pass_name: " << drr_pass_name
                       << " nice: " << nice
                       << " msg: " << drr_ctx.GetError().msg();
          }
        }
      }
    }
    return adt::Ok{};
  }

  adt::Result<DrrCtx> GetDrrCtx(const std::string& drr_pass_name,
                                const ap::registry::DrrRegistryItem& drr_item) {
    ADT_CHECK(drr_item->lambda->data.has_value());
    const auto& drr_func = drr_item->lambda->data.value();
    ADT_LET_CONST_REF(drr_ctx,
                      ApDrrHelper{}.Interpret(drr_func, drr_pass_name));
    if (!drr_ctx->pass_name.has_value()) {
      drr_ctx.shared_ptr()->pass_name = drr_pass_name;
    }
    return drr_ctx;
  }
};

adt::Result<ap::registry::Registry> TryGetRegistrySingleton() {
  ADT_LET_CONST_REF(registry, ApRegistryHelper{}.SingltonRegistry());
  return registry;
}

std::optional<ap::registry::Registry> GetRegistrySingleton() {
  const auto& registry = TryGetRegistrySingleton();
  if (registry.HasOkValue()) {
    return registry.GetOkValue();
  } else {
    LOG(ERROR) << "\nTraceback (most recent call last):\n"
               << registry.GetError().CallStackToString() << "\n"
               << registry.GetError().class_name() << ": "
               << registry.GetError().msg();
    return std::nullopt;
  }
}

}  // namespace

std::optional<std::unique_ptr<::pir::Pass>> CreateApLowerFusionOpPass() {
  if (GetRegistrySingleton().has_value()) {
    std::unique_ptr<::pir::Pass> pass = std::make_unique<ApLowerFusionOpPass>();
    return std::move(pass);
  } else {
    return std::nullopt;
  }
}

}  // namespace cinn::dialect::ir
