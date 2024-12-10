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

#include "paddle/ap/include/paddle/cinn/ap_lower_fusion_op_pass.h"

#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/atomic.h"
#include "paddle/ap/include/axpr/data_type_util.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/code_gen/arg_source_maker.h"
#include "paddle/ap/include/code_gen/matched_result_pattern_helper.h"
#include "paddle/ap/include/code_gen/value.h"
#include "paddle/ap/include/drr/drr_graph_descriptor.h"
#include "paddle/ap/include/drr/drr_node_descriptor.h"
#include "paddle/ap/include/drr/res_ptn_packed_ir_op_declare_data.h"
#include "paddle/ap/include/drr/result_pattern_helper.h"
#include "paddle/ap/include/drr/value.h"
#include "paddle/ap/include/graph/graph_helper.h"
#include "paddle/ap/include/index_expr/valid_index_expr_builder.h"
#include "paddle/ap/include/ir_match/graph_matcher.h"
#include "paddle/ap/include/ir_match/ir_match_ctx.h"
#include "paddle/ap/include/paddle/cinn/ap_drr_helper.h"
#include "paddle/ap/include/paddle/cinn/ap_kernel_define_helper.h"
#include "paddle/ap/include/paddle/cinn/ap_registry_helper.h"
#include "paddle/ap/include/paddle/indexed_ir_graph_util.h"
#include "paddle/ap/include/paddle/pir_graph_descriptor.h"
#include "paddle/ap/include/paddle/pir_node.h"
#include "paddle/ap/include/paddle/pir_node_descriptor.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace cinn::dialect::ir {

namespace adt = ap::adt;

namespace {

using ap::paddle::PirNode;

using DrrValue = ap::drr::Value;
using DrrNode = ap::drr::Node;

using DrrCtx = ap::drr::DrrCtx;

using DrrNativeIrValue = ap::drr::NativeIrValue<DrrNode>;
using DrrPackedIrValue = ap::drr::PackedIrValue<DrrNode>;
using DrrIrValue = ap::drr::IrValue;

using DrrNativeIrOp = ap::drr::NativeIrOp<DrrNode>;
using DrrNativeIrOpOperand = ap::drr::NativeIrOpOperand<DrrNode>;
using DrrNativeIrOpResult = ap::drr::NativeIrOpResult<DrrNode>;
using DrrPackedIrOp = ap::drr::PackedIrOp<DrrNode>;
using DrrPackedIrOpOperand = ap::drr::PackedIrOpOperand<DrrNode>;
using DrrPackedIrOpResult = ap::drr::PackedIrOpResult<DrrNode>;
using DrrOptPackedIrOp = ap::drr::OptPackedIrOp<DrrNode>;
using DrrOptPackedIrOpOperand = ap::drr::OptPackedIrOpOperand<DrrNode>;
using DrrOptPackedIrOpResult = ap::drr::OptPackedIrOpResult<DrrNode>;

using DrrIrOpImpl = std::variant<DrrNativeIrOp, DrrPackedIrOp>;

using IrMatchCtx = ap::ir_match::IrMatchCtx<PirNode>;

using ap::axpr::AnfExpr;
using CGValue = ap::code_gen::Value;
using CodeGenCtx = ap::code_gen::CodeGenCtx<PirNode>;
using CodeGenResult = ap::code_gen::CodeGenResult<CGValue>;
using ap::code_module::CodeModule;

struct DrrIrOp : public DrrIrOpImpl {
  using DrrIrOpImpl::DrrIrOpImpl;
  DEFINE_ADT_VARIANT_METHODS(DrrIrOpImpl);
};
using DrrGraphNode = ap::graph::Node<DrrNode>;
using GraphMatchCtx = ap::ir_match::GraphMatchCtx<PirNode>;

using PirNativeIrValue = ap::paddle::NativeIrValue;
using PirNativeIrOpOperand = ap::paddle::NativeIrOpOperand;
using PirNativeIrOpResult = ap::paddle::NativeIrOpResult;

adt::Result<DrrNode> GetApDrrDefaultAnchor(const DrrCtx& drr_ctx) {
  ADT_LET_CONST_REF(src_ptn_ctx, drr_ctx->GetSourcePatternCtx());
  auto ptn_node_area = src_ptn_ctx->node_arena;
  ap::graph::GraphDescriptor<DrrGraphNode, ap::drr::topo_kind::Default>
      source_pattern_graph{};
  ADT_CHECK(ptn_node_area->nodes().size() > 0);
  ap::graph::GraphHelper<DrrGraphNode, ap::drr::topo_kind::Default>
      graph_helper(source_pattern_graph);
  const auto& start_ptn_node = ptn_node_area->nodes().at(0).node();
  ADT_LET_CONST_REF(anchor_node, graph_helper.FindAnchor(start_ptn_node));
  ADT_LET_CONST_REF(default_anchor, anchor_node.Get());
  return default_anchor;
}

adt::Result<std::optional<DrrNativeIrOp>> GetApDrrNativeIrOpAnchor(
    const DrrCtx& drr_ctx) {
  ADT_LET_CONST_REF(src_ptn_ctx, drr_ctx->GetSourcePatternCtx());
  auto ptn_node_area = src_ptn_ctx->node_arena;
  ap::graph::GraphDescriptor<DrrGraphNode, ap::drr::topo_kind::Default>
      source_pattern_graph{};
  ADT_CHECK(ptn_node_area->nodes().size() > 0);
  ap::graph::GraphHelper<DrrGraphNode, ap::drr::topo_kind::Default>
      graph_helper(source_pattern_graph);
  const auto& start_ptn_node = ptn_node_area->nodes().at(0).node();
  auto IsNativeOpWithOutputs = [&](const auto& node) -> adt::Result<bool> {
    ADT_LET_CONST_REF(drr_node, node.Get());
    ADT_LET_CONST_REF(downstreams, node.DownstreamNodes());
    return drr_node.template Has<DrrNativeIrOp>() && downstreams.size() > 0;
  };
  const auto& Filter = IsNativeOpWithOutputs;
  ADT_LET_CONST_REF(anchor_node,
                    graph_helper.FilterAnchor(start_ptn_node, Filter));
  if (!anchor_node.has_value()) {
    return std::nullopt;
  }
  ADT_LET_CONST_REF(anchor, anchor_node.value().Get());
  ADT_LET_CONST_REF(native_ir_op, anchor.template TryGet<DrrNativeIrOp>());
  return native_ir_op;
}

adt::Result<std::vector<DrrIrValue>> GetResPtnOutputs(const DrrCtx& drr_ctx) {
  std::vector<DrrIrValue> ret;
  ADT_LET_CONST_REF(res_ptn_ctx, drr_ctx->GetResultPatternCtx());
  const auto& nodes = res_ptn_ctx->node_arena->nodes();
  for (const auto& drr_node : nodes) {
    ADT_LET_CONST_REF(downstreams, drr_node.node().DownstreamNodes());
    if (downstreams.size() == 0) {
      const auto& opt_drr_ir_value = DrrIrValue::OptCastFrom(drr_node);
      ADT_CHECK(opt_drr_ir_value.has_value());
      ret.push_back(opt_drr_ir_value.value());
    }
  }
  return ret;
}

struct ApLowerFusionOpPatternCtx {
  DrrCtx drr_ctx;
  std::vector<DrrIrValue> res_ptn_outputs;
  DrrNode default_anchor;
  std::optional<DrrNativeIrOp> native_op_anchor;
  std::string anchor_op_name;

  static adt::Result<ApLowerFusionOpPatternCtx> MakeFromDrrCtx(
      const DrrCtx& drr_ctx) {
    ADT_LET_CONST_REF(res_ptn_outputs, GetResPtnOutputs(drr_ctx));
    ADT_LET_CONST_REF(default_anchor, GetApDrrDefaultAnchor(drr_ctx));
    ADT_LET_CONST_REF(opt_native_ir_op_anchor,
                      GetApDrrNativeIrOpAnchor(drr_ctx));
    ADT_LET_CONST_REF(
        anchor_op_name,
        default_anchor.Match(
            [&](const DrrNativeIrOp& ir_op) -> adt::Result<std::string> {
              return ir_op->op_declare->op_name;
            },
            [&](const DrrPackedIrOp& ir_op) -> adt::Result<std::string> {
              return PirNode::GetOpNameFromDrrPackedOpName(
                  ir_op->op_declare->op_name);
            },
            [&](const DrrOptPackedIrOp& ir_op) -> adt::Result<std::string> {
              return PirNode::GetOpNameFromDrrPackedOpName(
                  ir_op->op_declare->op_name);
            },
            [&](const auto&) -> adt::Result<std::string> {
              return adt::errors::TypeError{
                  "default_anchor drr node should be a op node but value node "
                  "found."};
            }));
    return ApLowerFusionOpPatternCtx{drr_ctx,
                                     res_ptn_outputs,
                                     default_anchor,
                                     opt_native_ir_op_anchor,
                                     anchor_op_name};
  }
};

struct ApRewriter {
  ApLowerFusionOpPatternCtx ctx_;

  explicit ApRewriter(const ApLowerFusionOpPatternCtx& ctx) : ctx_(ctx) {}

  adt::Result<bool> Rewrite(const GraphMatchCtx& match_ctx,
                            pir::Operation* op,
                            pir::PatternRewriter* rewriter) const {
    ADT_CHECK(ctx_.drr_ctx->pass_name.has_value());
    LOG(ERROR) << "drr: " << ctx_.drr_ctx->pass_name.value() << " matched.";
    return RewriteByResultPattern(match_ctx, op->GetParent(), rewriter);
  }

 private:
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
    std::unordered_map<std::string, std::vector<pir::Value>> name2packed_values;

    adt::Result<std::size_t> GetMatchedOpOrderValue(pir::Operation* op) const {
      const auto iter = this->matched_op2order_value.find(op);
      if (iter == this->matched_op2order_value.end()) {
        return adt::errors::IndexError{
            "RewriteCtx::GetMatchedOpOrderValue failed."};
      }
      return iter->second;
    }

    adt::Result<pir::Value> GetNativeIrValue(
        const std::string& ir_value_name) const {
      const auto iter = this->name2native_value.find(ir_value_name);
      if (iter == this->name2native_value.end()) {
        return adt::errors::IndexError{
            "RewriteCtx::GetNativeIrValue() failed. key '" + ir_value_name +
            "' not found."};
      }
      return iter->second;
    }

    adt::Result<const std::vector<pir::Value>*> GetPackedIrValues(
        const std::string& ir_value_name) const {
      const auto iter = this->name2packed_values.find(ir_value_name);
      if (iter == this->name2packed_values.end()) {
        return adt::errors::IndexError{
            "RewriteCtx::GetPackedIrValues() failed. key '" + ir_value_name +
            "' not found"};
      }
      return &iter->second;
    }
  };

  adt::Result<std::unordered_set<pir::Operation*>> GetMatchedOps(
      const GraphMatchCtx& match_ctx) const {
    using DefaultDrrGraph =
        ap::graph::GraphDescriptor<DrrGraphNode, ap::drr::topo_kind::Default>;
    DefaultDrrGraph default_drr_graph{};
    ADT_LET_CONST_REF(src_ptn_ctx, ctx_.drr_ctx->GetSourcePatternCtx());
    const auto& nodes = src_ptn_ctx->node_arena->nodes();
    std::unordered_set<pir::Operation*> ops;
    for (const auto& drr_node : nodes) {
      ADT_LET_CONST_REF(is_op_node,
                        default_drr_graph.IsOpNode(drr_node.node()));
      if (!is_op_node) {
        continue;
      }
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
                to, rewrite_ctx.GetNativeIrValue(native_ir_value->name));
            return DoEachPair(from, to);
          },
          [&](const DrrPackedIrValue& packed_ir_value) -> adt::Result<adt::Ok> {
            ADT_LET_CONST_REF(from_nodes,
                              match_ctx->GetPackedBigGraphIrValueNodes(
                                  packed_ir_value->node));
            ADT_LET_CONST_REF(
                to_values_ptr,
                rewrite_ctx.GetPackedIrValues(packed_ir_value->name));
            ADT_CHECK(from_nodes->size() == to_values_ptr->size())
                << adt::errors::ValueError{
                       "from_nodes->size(): " +
                       std::to_string(from_nodes->size()) +
                       ", to_values_ptr->size(): " +
                       std::to_string(to_values_ptr->size()) + "."};
            for (int i = 0; i < from_nodes->size(); ++i) {
              const auto& from_node = from_nodes->at(i);
              ADT_LET_CONST_REF(
                  pir_value,
                  from_node.template TryGet<ap::paddle::NativeIrValue>());
              pir::Value from = pir_value.value;
              pir::Value to = to_values_ptr->at(i);
              ADT_RETURN_IF_ERR(DoEachPair(from, to));
            }
            return adt::Ok{};
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
    ADT_LET_CONST_REF(code_gen_result,
                      GetSerializedCodeGenResult(res_ptn_ir_op, match_ctx));
    const auto& [code_gen_lambda_str,
                 kernel_dispatch_func,
                 kernel_dispatch_const_data] = code_gen_result;
    ADT_LET_CONST_REF(infer_meta_lambda_str,
                      GetInferMetaLambdaStr(res_ptn_ir_op, match_ctx));
    ADT_LET_CONST_REF(kernel_dispatch_lambda_str,
                      GetKernelDispatchLambdaStr(kernel_dispatch_func));
    ADT_LET_CONST_REF(
        kernel_dispatch_const_data_lambda_str,
        GetKernelDispatchConstDataLambdaStr(
            res_ptn_ir_op, match_ctx, kernel_dispatch_const_data));
    ADT_LET_CONST_REF(num_outputs,
                      GetApKernelNumOutputs(res_ptn_ir_op, match_ctx));
    ADT_LET_CONST_REF(
        ap_pattern_fusion_combined_out,
        MakeApPatternFusionOp(rewriter,
                              new_ops,
                              combined_value,
                              num_outputs,
                              code_gen_lambda_str,
                              infer_meta_lambda_str,
                              kernel_dispatch_lambda_str,
                              kernel_dispatch_const_data_lambda_str));
    ADT_LET_CONST_REF(output_values,
                      GetPackedOpOutputValues(
                          rewriter, new_ops, ap_pattern_fusion_combined_out));
    ADT_RETURN_IF_ERR(UpdateApKernelOutputsInReplaceCtx(
        match_ctx, output_values, res_ptn_ir_op, rewrite_ctx));
    return adt::Ok{};
  }

  struct InputDimIndex {
    int input_idx;
    int tensor_axis;
  };

  struct OpInferMetaCtx {
    std::unordered_map<symbol::DimExpr, InputDimIndex> dim_expr2in_dim_index;
    mutable std::unordered_map<symbol::DimExpr, AnfExpr> dim_expr2anf_expr;
  };

  struct TensorMeta {
    std::vector<symbol::DimExpr> shape;
    pir::Type dtype;
  };

  adt::Result<std::string> GetInferMetaLambdaStr(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    ADT_LET_CONST_REF(infer_meta_ctx,
                      GetOpInferMetaCtx(res_ptn_ir_op, match_ctx));
    ADT_LET_CONST_REF(outputs, GetOpOutputPirValues(res_ptn_ir_op, match_ctx));
    auto ConstructLambdaBody =
        [&](ap::axpr::LetContext& ctx) -> adt::Result<AnfExpr> {
      for (int i = 0; i < outputs.size(); ++i) {
        const auto& output = outputs.at(i);
        auto& output_meta_var = ctx.Var("outputs").At(i);
        ADT_LET_CONST_REF(dim_exprs_ptr, GetShapeDimExprsPtrByValue(output));
        ADT_LET_CONST_REF(ddim_val,
                          ConstructDDims(&ctx, infer_meta_ctx, *dim_exprs_ptr));
        output_meta_var.SetAttr("dims", ddim_val);
        ADT_LET_CONST_REF(dtype, GetPirDataType(output));
        ADT_LET_CONST_REF(dtype_val,
                          ConstructDtype(&ctx, infer_meta_ctx, dtype));
        output_meta_var.SetAttr("dtype", dtype_val);
      }
      return ctx.None();
    };
    ap::axpr::LambdaExprBuilder lmbd;
    ADT_LET_CONST_REF(
        anf_expr, lmbd.TryLambda({"inputs", "outputs"}, ConstructLambdaBody));
    return anf_expr.DumpToJsonString();
  }

  adt::Result<pir::Type> GetPirDataType(pir::Value value) const {
    if (!value.type().isa<pir::DenseTensorType>()) {
      return adt::errors::NotImplementedError{
          "pir value must be of DenseTensorType"};
    }
    const auto dense_tensor_type =
        value.type().dyn_cast<pir::DenseTensorType>();
    return dense_tensor_type.dtype();
  }

  adt::Result<std::vector<pir::Value>> GetOpOutputPirValues(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    return GetMatchedPirOutputsOfRestPtnPackedIrOp(res_ptn_ir_op, match_ctx);
  }

  adt::Result<OpInferMetaCtx> GetOpInferMetaCtx(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    ADT_LET_CONST_REF(
        inputs,
        GetMatchedPirInputsOfRestPtnPackedIrOp(res_ptn_ir_op, match_ctx));
    OpInferMetaCtx infer_meta_ctx{};
    auto* map = &infer_meta_ctx.dim_expr2in_dim_index;
    for (int in_idx = 0; in_idx < inputs.size(); ++in_idx) {
      pir::Value input = inputs.at(in_idx);
      ADT_LET_CONST_REF(dim_exprs_ptr, GetShapeDimExprsPtrByValue(input));
      for (int tensor_axis = 0; tensor_axis < dim_exprs_ptr->size();
           ++tensor_axis) {
        const auto& dim_expr = dim_exprs_ptr->at(tensor_axis);
        map->emplace(dim_expr, InputDimIndex{in_idx, tensor_axis});
      }
    }
    return infer_meta_ctx;
  }

  adt::Result<const std::vector<symbol::DimExpr>*> GetShapeDimExprsPtrByValue(
      pir::Value value) const {
    auto* op = value.defining_op();
    ADT_CHECK(op != nullptr);
    auto* program = op->GetParentProgram();
    auto& shape_analysis = ::pir::ShapeAnalysisManager::Instance().Get(program);
    const auto& shape_or_data = shape_analysis.GetShapeOrDataForValue(value);
    using RetT = adt::Result<const std::vector<symbol::DimExpr>*>;
    return shape_or_data.Match(
        [&](const symbol::TensorShapeOrDataDimExprs& impl) -> RetT {
          return &impl.shape();
        },
        [&](const auto&) -> RetT {
          return adt::errors::TypeError{
              "GetShapeDimExprsPtrByValue only support "
              "TensorShapeOrDataDimExprs."};
        });
  }

  adt::Result<AnfExpr> ConstructDtype(ap::axpr::LetContext* ctx,
                                      const OpInferMetaCtx& infer_meta_ctx,
                                      pir::Type type) const {
    try {
      ::phi::DataType phi_dtype = ::paddle::dialect::TransToPhiDataType(type);
      ADT_LET_CONST_REF(dtype, ap::axpr::GetDataTypeFromPhiDataType(phi_dtype));
      return static_cast<AnfExpr>(ctx->Var("DataType").Attr(dtype.Name()));
    } catch (const std::exception& e) {
      return adt::errors::TypeError{
          "failed to cast from pir data type to phi data type."};
    }
  }

  adt::Result<AnfExpr> ConstructDDims(
      ap::axpr::LetContext* ctx,
      const OpInferMetaCtx& infer_meta_ctx,
      const std::vector<symbol::DimExpr>& dim_exprs) const {
    std::vector<AnfExpr> anf_dims;
    for (const auto& dim_expr : dim_exprs) {
      ADT_LET_CONST_REF(anf_dim_expr,
                        ConstructDDimDimExpr(ctx, infer_meta_ctx, dim_expr));
      anf_dims.emplace_back(anf_dim_expr);
    }
    return ctx->Call(ap::axpr::kBuiltinList(), anf_dims);
  }

  adt::Result<AnfExpr> ConstructDDimDimExpr(
      ap::axpr::LetContext* ctx,
      const OpInferMetaCtx& infer_meta_ctx,
      const symbol::DimExpr& dim_expr) const {
    return dim_expr.Match(
        [&](int64_t c) -> adt::Result<AnfExpr> { return ctx->Int64(c); },
        [&](const auto&) -> adt::Result<AnfExpr> {
          return ConstructDDimDimExprByInputs(ctx, infer_meta_ctx, dim_expr);
        });
  }

  adt::Result<AnfExpr> ConstructDDimDimExprByInputs(
      ap::axpr::LetContext* ctx,
      const OpInferMetaCtx& infer_meta_ctx,
      const symbol::DimExpr& dim_expr) const {
    const auto& idx_iter = infer_meta_ctx.dim_expr2in_dim_index.find(dim_expr);
    ADT_CHECK(idx_iter != infer_meta_ctx.dim_expr2in_dim_index.end());
    auto anf_expr_iter = infer_meta_ctx.dim_expr2anf_expr.find(dim_expr);
    if (anf_expr_iter == infer_meta_ctx.dim_expr2anf_expr.end()) {
      const auto& in_dim = ConstructInDimExpr(ctx, idx_iter->second);
      anf_expr_iter =
          infer_meta_ctx.dim_expr2anf_expr.emplace(dim_expr, in_dim).first;
    }
    return anf_expr_iter->second;
  }

  AnfExpr ConstructInDimExpr(ap::axpr::LetContext* ctx,
                             const InputDimIndex& idx) const {
    return static_cast<AnfExpr>(
        ctx->Var("inputs").At(idx.input_idx).Attr("dims").At(idx.tensor_axis));
  }

  adt::Result<AnfExpr> GetCodeFromBuiltinSerializableAttrMap(
      ap::axpr::LetContext* ctx,
      const ap::axpr::AttrMap<ap::axpr::SerializableValue>& attr_map) const {
    std::map<std::string, AnfExpr> kwargs;
    for (const auto& [keyword, val] : attr_map->storage) {
      ADT_LET_CONST_REF(val_anf,
                        GetCodeFromBuiltinSerializableAttrMapItem(ctx, val));
      kwargs[keyword] = val_anf;
    }
    return ctx->Apply("BuiltinSerializableAttrMap", {}, kwargs);
  }

  adt::Result<AnfExpr> GetCodeFromBuiltinSerializableAttrMapItem(
      ap::axpr::LetContext* ctx,
      const ap::axpr::SerializableValue& item) const {
    return item.Match(
        [&](const adt::Nothing&) -> adt::Result<AnfExpr> {
          return ctx->None();
        },
        [&](bool c) -> adt::Result<AnfExpr> { return ctx->Bool(c); },
        [&](int64_t c) -> adt::Result<AnfExpr> { return ctx->Int64(c); },
        [&](double c) -> adt::Result<AnfExpr> { return ctx->Double(c); },
        [&](const std::string& str) -> adt::Result<AnfExpr> {
          return ctx->String(str);
        },
        [&](const adt::List<ap::axpr::SerializableValue>& l)
            -> adt::Result<AnfExpr> {
          return GetCodeFromBuiltinSerializableAttrMapList(ctx, l);
        },
        [&](const ap::axpr::AttrMap<ap::axpr::SerializableValue>& object)
            -> adt::Result<AnfExpr> {
          return GetCodeFromBuiltinSerializableAttrMap(ctx, object);
        },
        [&](const ap::axpr::Function<ap::axpr::SerializableValue>& function)
            -> adt::Result<AnfExpr> {
          const auto& lambda = function->lambda;
          const AnfExpr& anf_expr = ap::axpr::ConvertCoreExprToAnfExpr(lambda);
          AnfExpr ret{ctx->Attr(anf_expr, "__function__")};
          return ret;
        },
        [&](const auto&) -> adt::Result<AnfExpr> {
          std::ostringstream ss;
          ss << "Builtin serializable types are: NoneType, bool, int, float, "
                "str, function_code, list, BuiltinSerializableAttrMap (not "
                "include '"
             << ap::axpr::GetTypeName(item.template CastTo<CGValue>()) << "').";
          return adt::errors::ValueError{ss.str()};
        });
  }

  adt::Result<AnfExpr> GetCodeFromBuiltinSerializableAttrMapList(
      ap::axpr::LetContext* ctx,
      const adt::List<ap::axpr::SerializableValue>& list) const {
    std::vector<AnfExpr> elt_anf_exprs;
    for (const auto& elt : *list) {
      ADT_LET_CONST_REF(elt_anf_expr,
                        GetCodeFromBuiltinSerializableAttrMapItem(ctx, elt));
      elt_anf_exprs.emplace_back(elt_anf_expr);
    }
    return ctx->Call(ap::axpr::kBuiltinList(), elt_anf_exprs);
  }

  adt::Result<std::string> GetKernelDispatchConstDataLambdaStr(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx,
      const ap::axpr::AttrMap<ap::axpr::SerializableValue>&
          kernel_dispatch_const_data) const {
    ap::axpr::LambdaExprBuilder lmbd;
    auto ConstructLambdaBody = [&](auto& ctx) -> adt::Result<AnfExpr> {
      ADT_LET_CONST_REF(data,
                        GetCodeFromBuiltinSerializableAttrMap(
                            &ctx, kernel_dispatch_const_data));
      return data;
    };
    ADT_LET_CONST_REF(anf_expr, lmbd.TryLambda({}, ConstructLambdaBody));
    return anf_expr.DumpToJsonString();
  }

  struct SerializedCodeGenResult {
    std::string code_gen_lambda_str;
    ap::axpr::Function<ap::axpr::SerializableValue> kernel_dispatch_func;
    ap::axpr::AttrMap<ap::axpr::SerializableValue> kernel_dispatch_const_data;
  };

  adt::Result<SerializedCodeGenResult> GetSerializedCodeGenResult(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    const auto& op_declare = res_ptn_ir_op->op_declare;
    ADT_LET_CONST_REF(
        op_declare_data,
        op_declare->cast_data<ap::drr::ResPtnPackedIrOpDeclareData>());
    const auto& lambda = op_declare_data->code_gen_func();
    ADT_LET_CONST_REF(code_gen_result,
                      GetApKernelModule(lambda, match_ctx, res_ptn_ir_op));
    ADT_LET_CONST_REF(
        anf_expr, ConvertApKernelModuleToAnfExpr(code_gen_result->code_module));
    const std::string& code_gen_lambda_str = anf_expr.DumpToJsonString();
    const auto& kernel_dispatch_func = code_gen_result->kernel_dispatch_func;
    auto* data = &code_gen_result.shared_ptr()->kernel_dispatch_const_data;
    ADT_RETURN_IF_ERR(
        InsertApKernelInputIndexOrSlices(data, res_ptn_ir_op, match_ctx));
    ADT_RETURN_IF_ERR(
        InsertApKernelOutputIndexOrSlices(data, res_ptn_ir_op, match_ctx));
    ADT_RETURN_IF_ERR(
        InsertApKernelInputName2Index(data, res_ptn_ir_op, match_ctx));
    ADT_RETURN_IF_ERR(
        InsertApKernelOutputName2Index(data, res_ptn_ir_op, match_ctx));
    return SerializedCodeGenResult{
        code_gen_lambda_str, kernel_dispatch_func, *data};
  }

  adt::Result<adt::Ok> InsertApKernelInputIndexOrSlices(
      ap::axpr::AttrMap<ap::axpr::SerializableValue>* object,
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    adt::List<ap::axpr::SerializableValue> list;
    using Ok = adt::Result<adt::Ok>;
    auto DoEachIndex = [&](int64_t idx) -> Ok {
      list->emplace_back(idx);
      return adt::Ok{};
    };
    auto DoEachSlice = [&](int64_t start, int64_t end) -> Ok {
      adt::List<ap::axpr::SerializableValue> range{start, end};
      list->emplace_back(range);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitApKernelInputIndexOrSlice(
        res_ptn_ir_op, match_ctx, DoEachIndex, DoEachSlice));
    ADT_CHECK(
        (*object)->Emplace("__builtin_ap_kernel_input_indexes_slices", list));
    return adt::Ok{};
  }

  adt::Result<adt::Ok> InsertApKernelOutputIndexOrSlices(
      ap::axpr::AttrMap<ap::axpr::SerializableValue>* object,
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    adt::List<ap::axpr::SerializableValue> list;
    using Ok = adt::Result<adt::Ok>;
    auto DoEachIndex = [&](int64_t idx) -> Ok {
      list->emplace_back(idx);
      return adt::Ok{};
    };
    auto DoEachSlice = [&](int64_t start, int64_t end) -> Ok {
      adt::List<ap::axpr::SerializableValue> range{start, end};
      list->emplace_back(range);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitApKernelOutputIndexOrSlice(
        res_ptn_ir_op, match_ctx, DoEachIndex, DoEachSlice));
    ADT_CHECK(
        (*object)->Emplace("__builtin_ap_kernel_output_indexes_slices", list));
    return adt::Ok{};
  }

  adt::Result<CodeGenResult> GetApKernelModule(
      const ap::axpr::Function<ap::axpr::SerializableValue>& lambda,
      const GraphMatchCtx& match_ctx,
      const DrrPackedIrOp& res_ptn_ir_op) const {
    ADT_LET_CONST_REF(src_ptn_ctx, ctx_.drr_ctx->GetSourcePatternCtx());
    IrMatchCtx ir_match_ctx{src_ptn_ctx, match_ctx};
    ADT_LET_CONST_REF(arg_source_ctx,
                      MakeArgSourceCtx(match_ctx, res_ptn_ir_op));
    CodeGenCtx code_gen_ctx{ir_match_ctx, res_ptn_ir_op, arg_source_ctx};
    ApKernelDefineHelper helper{};
    ADT_LET_CONST_REF(result, helper.Interpret(lambda, code_gen_ctx));
    return result;
  }

  adt::Result<ap::code_gen::ArgSourceCtx<PirNode>> MakeArgSourceCtx(
      const GraphMatchCtx& match_ctx,
      const DrrPackedIrOp& res_ptn_ir_op) const {
    ap::code_gen::MatchedResultPatternHelper<PirNode> helper{match_ctx,
                                                             ctx_.drr_ctx};
    ap::code_gen::ArgSourceMaker<PirNode> maker{helper};
    ADT_LET_CONST_REF(arg_source_ctx, maker.MakeArgSourceCtx(res_ptn_ir_op));
    return arg_source_ctx;
  }

  adt::Result<AnfExpr> ConvertApKernelModuleToAnfExpr(
      const CodeModule& m) const {
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
    auto ConvertCudaKernelSourceCodeConstruction =
        [&](auto& ctx, const auto& cuda_kernel) -> AnfExpr {
      const auto& str = ctx.String(cuda_kernel->source_code);
      return ctx.Call("CudaKernelSourceCode", str);
    };
    auto ConvertSourceCodeConstruction =
        [&](auto& ctx) -> adt::Result<AnfExpr> {
      return m->source_code.Match(
          [&](const ap::code_module::CudaKernelSourceCode& cuda_kernel)
              -> adt::Result<AnfExpr> {
            return ConvertCudaKernelSourceCodeConstruction(ctx, cuda_kernel);
          },
          [&](const ap::code_module::Project& project) -> adt::Result<AnfExpr> {
            return ConvertProjectConstruct(&ctx, project);
          });
    };
    auto ConstructLambdaBody = [&](auto& ctx) -> adt::Result<AnfExpr> {
      const auto& declare = ConvertFuncDeclareList(ctx);
      ADT_LET_CONST_REF(source_code, ConvertSourceCodeConstruction(ctx));
      return ctx.Call("CodeModule", declare, source_code);
    };
    return ap::axpr::LambdaExprBuilder{}.TryLambda({}, ConstructLambdaBody);
  }

  adt::Result<AnfExpr> ConvertProjectConstruct(
      ap::axpr::LetContext* ctx,
      const ap::code_module::Project& project) const {
    const auto& attrs = project->others;
    ADT_LET_CONST_REF(others_anf_expr,
                      GetCodeFromBuiltinSerializableAttrMap(ctx, attrs));
    return ctx->Apply(
        "Project",
        {},
        {
            {"nested_files",
             ConvertProjectNestedFiles(ctx, project->nested_files)},
            {"cmd", AnfExpr{ctx->String(project->cmd)}},
            {"so_relative_path", AnfExpr{ctx->String(project->cmd)}},
            {"others", others_anf_expr},
        });
  }

  AnfExpr ConvertProjectNestedFiles(ap::axpr::LetContext* ctx,
                                    const ap::code_module::File& file) const {
    return file.Match(
        [&](const ap::code_module::FileContent& file_content) -> AnfExpr {
          const auto& str = file_content->file_content;
          return ctx->Var("Project").Attr("FileContent").Call(ctx->String(str));
        },
        [&](const ap::code_module::SoftLink& soft_link) -> AnfExpr {
          const auto& str = soft_link->linked_file_relative_path;
          return ctx->Var("Project").Attr("SoftLink").Call(ctx->String(str));
        },
        [&](const ap::code_module::Directory<ap::code_module::File>& dir)
            -> AnfExpr {
          std::map<std::string, AnfExpr> kwargs;
          for (const auto& [k, v] : dir.dentry2file->storage) {
            kwargs[k] = ConvertProjectNestedFiles(ctx, v);
          }
          return ctx->Apply(ctx->Var("Project").Attr("Directory"), {}, kwargs);
        });
  }

  adt::Result<std::string> GetKernelDispatchLambdaStr(
      const ap::axpr::Function<ap::axpr::SerializableValue>&
          kernel_dispatch_func) const {
    const auto& lambda = kernel_dispatch_func->lambda;
    ap::axpr::AnfExpr anf_expr = ap::axpr::ConvertCoreExprToAnfExpr(lambda);
    return anf_expr.DumpToJsonString();
  }

  adt::Result<pir::Value> MakeApPatternFusionOp(
      pir::PatternRewriter* rewriter,
      std::set<pir::Operation*>* new_ops,
      pir::Value input,
      std::size_t num_outputs,
      const std::string& code_gen_lambda_str,
      const std::string& infer_meta_lambda_str,
      const std::string& kernel_dispatch_lambda_str,
      const std::string& kernel_dispatch_const_data_lambda_str) const {
    auto ap_unary = rewriter->Build<paddle::dialect::ApUnaryOp>(
        input,
        num_outputs,
        code_gen_lambda_str,
        infer_meta_lambda_str,
        kernel_dispatch_lambda_str,
        kernel_dispatch_const_data_lambda_str);
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
      const GraphMatchCtx& match_ctx,
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
            ADT_CHECK(rewrite_ctx->name2packed_values.emplace(k, v).second);
            return adt::Ok{};
          });
    };
    ADT_RETURN_IF_ERR(VisitEachMatchedDrrIrValueAndOutputSlice(
        match_ctx, output_values, res_ptn_ir_op, UpdateRewriteCtx));
    return adt::Ok{};
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitEachMatchedDrrIrValueAndOutputSlice(
      const GraphMatchCtx& match_ctx,
      const std::vector<pir::Value>& output_values,
      const DrrPackedIrOp& res_ptn_ir_op,
      const DoEachT& DoEach) const {
    ap::code_gen::MatchedResultPatternHelper<PirNode> helper{match_ctx,
                                                             ctx_.drr_ctx};
    return helper.VisitEachMatchedDrrIrValueAndOutputSlice<pir::Value>(
        output_values, res_ptn_ir_op, DoEach);
  }

  adt::Result<std::size_t> GetResPtnNumPirValues(
      const DrrIrValue& drr_ir_value, const GraphMatchCtx& match_ctx) const {
    ap::code_gen::MatchedResultPatternHelper<PirNode> helper{match_ctx,
                                                             ctx_.drr_ctx};
    return helper.GetResPtnNumBirValues(drr_ir_value);
  }

  adt::Result<std::size_t> GetApKernelNumOutputs(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    ap::code_gen::MatchedResultPatternHelper<PirNode> helper{match_ctx,
                                                             ctx_.drr_ctx};
    return helper.GetApKernelNumOutputs(res_ptn_ir_op);
  }

  template <typename DoEachIndexT, typename DoEachSliceT>
  adt::Result<adt::Ok> VisitApKernelInputIndexOrSlice(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx,
      const DoEachIndexT& DoEachIndex,
      const DoEachSliceT& DoEachSlice) const {
    ap::code_gen::MatchedResultPatternHelper<PirNode> helper{match_ctx,
                                                             ctx_.drr_ctx};
    return helper.VisitApKernelInputIndexOrSlice(
        res_ptn_ir_op, DoEachIndex, DoEachSlice);
  }

  template <typename DoEachIndexT, typename DoEachSliceT>
  adt::Result<adt::Ok> VisitApKernelOutputIndexOrSlice(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx,
      const DoEachIndexT& DoEachIndex,
      const DoEachSliceT& DoEachSlice) const {
    ap::code_gen::MatchedResultPatternHelper<PirNode> helper{match_ctx,
                                                             ctx_.drr_ctx};
    return helper.VisitApKernelOutputIndexOrSlice(
        res_ptn_ir_op, DoEachIndex, DoEachSlice);
  }

  adt::Result<adt::Ok> InsertApKernelInputName2Index(
      ap::axpr::AttrMap<ap::axpr::SerializableValue>* object,
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    ap::axpr::AttrMap<ap::axpr::SerializableValue> name2idx;
    int64_t idx = 0;
    auto DoEachIrValue =
        [&](const DrrIrValue& drr_ir_value) -> adt::Result<adt::Ok> {
      ADT_CHECK(name2idx->Emplace(drr_ir_value.name(), idx));
      ++idx;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(
        VisitResPtnInputIrValueByResPtnIrOp(res_ptn_ir_op, DoEachIrValue));
    ADT_CHECK((*object)->Emplace("__builtin_ap_kernel_input_name_to_index",
                                 name2idx));
    return adt::Ok{};
  }

  adt::Result<adt::Ok> InsertApKernelOutputName2Index(
      ap::axpr::AttrMap<ap::axpr::SerializableValue>* object,
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    ap::axpr::AttrMap<ap::axpr::SerializableValue> name2idx;
    int64_t idx = 0;
    auto DoEachIrValue =
        [&](const DrrIrValue& drr_ir_value) -> adt::Result<adt::Ok> {
      ADT_CHECK(name2idx->Emplace(drr_ir_value.name(), idx));
      ++idx;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(
        VisitResPtnOutputIrValueByResPtnIrOp(res_ptn_ir_op, DoEachIrValue));
    ADT_CHECK((*object)->Emplace("__builtin_ap_kernel_output_name_to_index",
                                 name2idx));
    return adt::Ok{};
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
    ap::drr::ResultPatternHelper helper{ctx_.drr_ctx};
    return helper.VisitResPtnInputIrValueByResPtnIrOp(res_ptn_ir_op, DoEach);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitResPtnOutputIrValueByResPtnIrOp(
      const DrrPackedIrOp& res_ptn_ir_op, const DoEachT& DoEach) const {
    ap::drr::ResultPatternHelper helper{ctx_.drr_ctx};
    return helper.VisitResPtnOutputIrValueByResPtnIrOp(res_ptn_ir_op, DoEach);
  }

  std::optional<DrrIrValue> SrcPtnIrValue4ResPtnIrValue(
      const DrrIrValue& res_ptn_ir_value) const {
    ap::drr::ResultPatternHelper helper{ctx_.drr_ctx};
    return helper.SrcPtnIrValue4ResPtnIrValue(res_ptn_ir_value);
  }

  adt::Result<adt::Ok> InsertInputPirValueToReplaceCtx(
      const DrrPackedIrOp& res_ptn_ir_op,
      RewriteCtx* rewrite_ctx,
      const GraphMatchCtx& match_ctx) const {
    using Ok = adt::Result<adt::Ok>;
    auto InitInput = [&](const DrrIrValue& drr_ir_value) -> Ok {
      return drr_ir_value.Match(
          [&](const DrrNativeIrValue& res_ptn_ir_value) -> Ok {
            ADT_RETURN_IF_ERR(InsertNativeIrValueToReplaceCtx(
                res_ptn_ir_value, rewrite_ctx, match_ctx));
            return adt::Ok{};
          },
          [&](const DrrPackedIrValue& res_ptn_ir_value) -> Ok {
            ADT_RETURN_IF_ERR(InsertPackedIrValueToReplaceCtx(
                res_ptn_ir_value, rewrite_ctx, match_ctx));
            return adt::Ok{};
          });
    };
    ADT_RETURN_IF_ERR(
        VisitResPtnInputIrValueByResPtnIrOp(res_ptn_ir_op, InitInput));
    return adt::Ok{};
  }

  adt::Result<adt::Ok> InsertNativeIrValueToReplaceCtx(
      const DrrNativeIrValue& res_ptn_ir_value,
      RewriteCtx* rewrite_ctx,
      const GraphMatchCtx& match_ctx) const {
    const auto iter =
        rewrite_ctx->name2native_value.find(res_ptn_ir_value->name);
    if (iter != rewrite_ctx->name2native_value.end()) {
      return adt::Ok{};
    }
    const auto& opt_ir_value = SrcPtnIrValue4ResPtnIrValue(res_ptn_ir_value);
    ADT_CHECK(opt_ir_value.has_value());
    const auto& ir_value = opt_ir_value.value();
    ADT_LET_CONST_REF(pir_node,
                      match_ctx->GetSoleBigGraphNode(ir_value.node()));
    ADT_LET_CONST_REF(pir_value,
                      pir_node.template TryGet<ap::paddle::NativeIrValue>())
        << adt::errors::TypeError{
               "pir_node is not an ap::paddle::NativeIrValue"};
    rewrite_ctx->name2native_value[ir_value.name()] = pir_value.value;
    return adt::Ok{};
  }

  adt::Result<adt::Ok> InsertPackedIrValueToReplaceCtx(
      const DrrPackedIrValue& res_ptn_ir_value,
      RewriteCtx* rewrite_ctx,
      const GraphMatchCtx& match_ctx) const {
    using Ok = adt::Result<adt::Ok>;
    const auto iter =
        rewrite_ctx->name2packed_values.find(res_ptn_ir_value->name);
    if (iter != rewrite_ctx->name2packed_values.end()) {
      return adt::Ok{};
    }
    const auto& opt_ir_value = SrcPtnIrValue4ResPtnIrValue(res_ptn_ir_value);
    ADT_CHECK(opt_ir_value.has_value());
    const auto& ir_value = opt_ir_value.value();
    auto* vec = &rewrite_ctx->name2packed_values[ir_value.name()];
    ADT_CHECK(vec->empty());
    auto AppendNode = [&](const PirNode& pir_node) -> Ok {
      ADT_LET_CONST_REF(pir_value,
                        pir_node.template TryGet<ap::paddle::NativeIrValue>())
          << adt::errors::TypeError{
                 "pir_node is not an ap::paddle::NativeIrValue"};
      vec->emplace_back(pir_value.value);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(
        match_ctx->VisitPackedBigGraphIrValueNode(ir_value.node(), AppendNode));
    return adt::Ok{};
  }

  adt::Result<PirNativeIrValue> CastToPirNativeIrValue(
      const PirNode& pir_node) const {
    using RetT = adt::Result<PirNativeIrValue>;
    return pir_node.Match(
        [&](const typename PirNode::native_value_type& bir_value) -> RetT {
          return bir_value;
        },
        [&](const typename PirNode::ref_value_type& ref_value) -> RetT {
          return ref_value.GetOwnerNativeIrValue();
        },
        [&](const auto&) -> RetT {
          return adt::errors::TypeError{
              "pir_node is not an PirNode::native_value_type or "
              "PirNode::ref_value_type"};
        });
  }

  adt::Result<std::vector<pir::Value>> GetMatchedPirInputsOfRestPtnPackedIrOp(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    std::vector<pir::Value> ret;
    auto CollectInput = [&](const PirNode& pir_node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(pir_value, CastToPirNativeIrValue(pir_node));
      ret.emplace_back(pir_value.value);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitMatchedPirInputOfRestPtnPackedIrOp(
        res_ptn_ir_op, match_ctx, CollectInput));
    return ret;
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitMatchedPirInputOfRestPtnPackedIrOp(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx,
      const DoEachT& DoEach) const {
    ap::code_gen::MatchedResultPatternHelper<PirNode> helper{match_ctx,
                                                             ctx_.drr_ctx};
    return helper.VisitMatchedBirInputOfRestPtnPackedIrOp(res_ptn_ir_op,
                                                          DoEach);
  }

  adt::Result<std::vector<pir::Value>> GetMatchedPirOutputsOfRestPtnPackedIrOp(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    std::vector<pir::Value> ret;
    using Ok = adt::Result<adt::Ok>;
    auto CollectOutput = [&](const PirNode& pir_node) -> Ok {
      ADT_LET_CONST_REF(pir_value, CastToPirNativeIrValue(pir_node));
      ret.emplace_back(pir_value.value);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitMatchedPirOutputOfRestPtnPackedIrOp(
        res_ptn_ir_op, match_ctx, CollectOutput));
    return ret;
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitMatchedPirOutputOfRestPtnPackedIrOp(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx,
      const DoEachT& DoEach) const {
    ap::code_gen::MatchedResultPatternHelper<PirNode> helper{match_ctx,
                                                             ctx_.drr_ctx};
    return helper.VisitMatchedBirOutputOfRestPtnPackedIrOp(res_ptn_ir_op,
                                                           DoEach);
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
          const auto& iter = rewrite_ctx.name2packed_values.find(name);
          ADT_CHECK(iter != rewrite_ctx.name2packed_values.end());
          for (const auto& value : iter->second) {
            ADT_RETURN_IF_ERR(DoEach(value));
          }
          return adt::Ok{};
        }));
    return adt::Ok{};
  }
};

class NativeOpAnchorApLowerFusionOpPattern : public pir::RewritePattern {
 private:
  ApLowerFusionOpPatternCtx ctx_;
  ApRewriter ap_rewriter_;
  mutable std::unordered_set<pir::Operation*> rewrited_;

 public:
  NativeOpAnchorApLowerFusionOpPattern(pir::IrContext* ir_context,
                                       const ApLowerFusionOpPatternCtx& ctx)
      : pir::RewritePattern(ctx.anchor_op_name, 1, ir_context, {}),
        ctx_(ctx),
        ap_rewriter_(ctx) {}

  bool MatchAndRewrite(
      pir::Operation* op,
      pir::PatternRewriter& rewriter) const override {  // // NOLINT
    if (rewrited_.count(op) > 0) {
      return false;
    }
    const auto& ret = TryMatchAndRewrite(op, &rewriter);
    if (ret.HasError()) {
      LOG(ERROR) << "\nTraceback (most recent call last):\n"
                 << ret.GetError().CallStackToString() << "\n"
                 << ret.GetError().class_name() << ": " << ret.GetError().msg();
      return false;
    }
    LOG(ERROR) << "MatchAndRewrite: op: " << op
               << ", ret: " << ret.GetOkValue();
    rewrited_.insert(op);
    return ret.GetOkValue();
  }

  adt::Result<bool> TryMatchAndRewrite(pir::Operation* op,
                                       pir::PatternRewriter* rewriter) const {
    ADT_LET_CONST_REF(match_ctx, GetMatchCtx(op));
    ADT_CHECK(ctx_.drr_ctx->pass_name.has_value());
    LOG(ERROR) << "drr: " << ctx_.drr_ctx->pass_name.value() << " matched.";
    return ap_rewriter_.Rewrite(match_ctx, op, rewriter);
  }

  template <typename NodeT>
  using NativeORGraph =
      ap::graph::GraphDescriptor<NodeT,
                                 ap::drr::topo_kind::NativeOperandAndResult>;

  template <typename NodeT>
  using DefaultGraph =
      ap::graph::GraphDescriptor<NodeT, ap::drr::topo_kind::Default>;

  template <typename NodeT>
  using RefAugmentedGraph =
      ap::graph::GraphDescriptor<NodeT, ap::drr::topo_kind::RefAugmented>;

  adt::Result<GraphMatchCtx> GetMatchCtx(pir::Operation* op) const {
    DefaultGraph<DrrGraphNode> drr_graph{};
    DefaultGraph<PirNode> pir_graph{};
    auto* parent_block = op->GetParent();
    ADT_CHECK(parent_block != nullptr);
    auto* parent_op = parent_block->GetParentOp();
    ADT_CHECK(!parent_op->isa<cinn::dialect::FusionOp>());
    ADT_CHECK(ctx_.native_op_anchor.has_value());
    const auto& native_op_anchor = ctx_.native_op_anchor.value();
    {
      ADT_LET_CONST_REF(
          anchor_cstr, drr_graph.GetSmallGraphNodeCstr(native_op_anchor->node));
      ap::paddle::NativeIrOp native_ir_op{op};
      ADT_LET_CONST_REF(satisfy_constraint,
                        pir_graph.Satisfy(native_ir_op, anchor_cstr));
      ADT_CHECK(satisfy_constraint) << adt::errors::ValueError{
          "pir_graph.Satisfy(native_ir_op, anchor_cstr) test failed."};
    }
    ADT_LET_CONST_REF(drr_op_result_anchor,
                      GetFirstNativeDrrIrOpResult(native_op_anchor));
    ADT_LET_CONST_REF(pir_op_result_anchor, GetFirstNativePirIrOpResult(op));
    {
      ADT_LET_CONST_REF(drr_op_result_anchor_cstr,
                        drr_graph.GetSmallGraphNodeCstr(drr_op_result_anchor));
      ADT_LET_CONST_REF(
          satisfy_constraint,
          pir_graph.Satisfy(pir_op_result_anchor, drr_op_result_anchor_cstr));
      ADT_CHECK(satisfy_constraint) << adt::errors::ValueError{
          std::string() +
          "pir_graph.Satisfy(pir_op_result_anchor, drr_op_result_anchor_cstr) "
          "test failed. pir_op_result_anchor: " +
          DebugId(pir_op_result_anchor) +
          ", drr_op_result_anchor: " + DebugId(drr_op_result_anchor) +
          ", pir_op: " + DebugId(ap::paddle::NativeIrOp{op}) +
          ", drr_native_op: " + DebugId(native_op_anchor->node) + "."};
    }
    std::optional<GraphMatchCtx> opt_graph_match_ctx;
    {
      NativeORGraph<PirNode> pir_native_operand_result_graph{};
      NativeORGraph<DrrGraphNode> drr_native_operand_result_graph{};
      using NativeOR = ap::drr::topo_kind::NativeOperandAndResult;
      ap::ir_match::GraphMatcher<PirNode, NativeOR, NativeOR> graph_matcher(
          pir_native_operand_result_graph, drr_native_operand_result_graph);
      ADT_LET_CONST_REF(graph_ctx,
                        graph_matcher.MatchByAnchor(pir_op_result_anchor,
                                                    drr_op_result_anchor));
      opt_graph_match_ctx = graph_ctx;
      ADT_LET_CONST_REF(graph_matched,
                        graph_matcher.IsGraphMatched(
                            opt_graph_match_ctx.value(), drr_op_result_anchor));
      ADT_CHECK(graph_matched) << adt::errors::MismatchError{};
    }
    ADT_CHECK(opt_graph_match_ctx.has_value());
    {
      ADT_LET_CONST_REF(
          ref_match_ctx,
          GetRefMatchCtx(opt_graph_match_ctx.value(), drr_op_result_anchor));
      RefAugmentedGraph<PirNode> pir_augmented_graph{ref_match_ctx};
      using RefAugmented = ap::drr::topo_kind::RefAugmented;
      using Default = ap::drr::topo_kind::Default;
      ap::ir_match::GraphMatcher<PirNode, RefAugmented, Default> graph_matcher(
          pir_augmented_graph, drr_graph);
      ADT_RETURN_IF_ERR(graph_matcher.UpdateByConnectionsUntilDone(
          &opt_graph_match_ctx.value(), drr_op_result_anchor));
      ADT_LET_CONST_REF(graph_matched,
                        graph_matcher.IsGraphMatched(
                            opt_graph_match_ctx.value(), drr_op_result_anchor));
      ADT_CHECK(graph_matched);
    }
    return opt_graph_match_ctx.value();
  }

  std::string DebugId(const PirNode& pir_node) const {
    return ap::graph::NodeDescriptor<PirNode>{}.DebugId(pir_node);
  }

  std::string DebugId(const DrrGraphNode& drr_node) const {
    return ap::graph::NodeDescriptor<DrrGraphNode>{}.DebugId(drr_node);
  }

  template <typename NodeT>
  using AllOperandAndResultGraph =
      ap::graph::GraphDescriptor<NodeT,
                                 ap::drr::topo_kind::AllOperandAndResult>;

  using RefNodeInfo =
      ap::ir_match::RefNodeInfo<PirNativeIrValue, PirNativeIrOpOperand>;
  using RefMatchCtx =
      ap::ir_match::RefMatchCtx<PirNativeIrValue, PirNativeIrOpOperand>;

  adt::Result<RefMatchCtx> GetRefMatchCtx(const GraphMatchCtx& graph_match_ctx,
                                          const DrrGraphNode& anchor) const {
    AllOperandAndResultGraph<PirNode> pir_graph{};
    AllOperandAndResultGraph<DrrGraphNode> drr_graph{};
    using AllOR = ap::drr::topo_kind::AllOperandAndResult;
    ap::ir_match::GraphMatcher<PirNode, AllOR, AllOR> graph_matcher(pir_graph,
                                                                    drr_graph);
    RefMatchCtx ref_match_ctx{};
    using Ok = adt::Result<adt::Ok>;
    auto DoEachMismatched = [&](const DrrGraphNode& node) -> Ok {
      ADT_LET_CONST_REF(drr_node, node.Get());
      return drr_node.Match(
          [&](const DrrOptPackedIrOpResult& op_result) -> Ok {
            ADT_LET_CONST_REF(ref_node_info,
                              GetRefNodeInfo(graph_match_ctx, op_result));
            if (ref_node_info.has_value()) {
              ADT_RETURN_IF_ERR(
                  ref_match_ctx->AddRefNodeInfo(ref_node_info.value()));
            }
            return adt::Ok{};
          },
          [&](const DrrOptPackedIrOpOperand& impl) -> Ok {
            // do nothing.
            return adt::Ok{};
          },
          [&](const DrrPackedIrOpOperand& impl) -> Ok {
            // do nothing.
            return adt::Ok{};
          },
          [&](const DrrPackedIrOpResult& impl) -> Ok {
            // do nothing.
            return adt::Ok{};
          },
          [&](const auto& impl) -> Ok {
            const char* type_name = typeid(std::decay_t<decltype(impl)>).name();
            return adt::errors::ValueError{
                std::string() +
                "GetRefValue2Operands unexpected mismatched DrrGraphNode: " +
                type_name};
          });
    };
    ADT_RETURN_IF_ERR(graph_matcher.VisitMisMatchedNodes(
        graph_match_ctx, anchor, DoEachMismatched));
    return ref_match_ctx;
  }

  adt::Result<std::optional<RefNodeInfo>> GetRefNodeInfo(
      const GraphMatchCtx& graph_match_ctx,
      const DrrOptPackedIrOpResult& op_result) const {
    ADT_LET_CONST_REF(opt_inner_ref_node_info,
                      GetInnerRefNodeInfo(graph_match_ctx, op_result));
    if (opt_inner_ref_node_info.has_value()) {
      return opt_inner_ref_node_info.value();
    }
    ADT_LET_CONST_REF(opt_output_ref_node_info,
                      GetOutputRefNodeInfo(graph_match_ctx, op_result));
    if (opt_output_ref_node_info.has_value()) {
      return opt_output_ref_node_info.value();
    }
    ADT_LET_CONST_REF(opt_input_ref_node_info,
                      GetInputRefNodeInfo(graph_match_ctx, op_result));
    if (opt_input_ref_node_info.has_value()) {
      return opt_input_ref_node_info.value();
    }
    return std::nullopt;
  }

  adt::Result<std::optional<RefNodeInfo>> GetInnerRefNodeInfo(
      const GraphMatchCtx& graph_match_ctx,
      const DrrOptPackedIrOpResult& drr_op_result) const {
    DefaultGraph<PirNode> default_pir_graph{};
    AllOperandAndResultGraph<DrrGraphNode> all_o_r_drr_graph{};
    const auto& topo_match_ctx = graph_match_ctx->topo_match_ctx;
    ADT_LET_CONST_REF(
        drr_op_operand,
        all_o_r_drr_graph.CastSoleUnignoredInput<DrrOptPackedIrOpOperand>(
            drr_op_result));
    {
      ADT_LET_CONST_REF(num_drr_op_result_downstreams,
                        all_o_r_drr_graph.GetNumOutputs(drr_op_result));
      if (num_drr_op_result_downstreams == 0) {
        return std::nullopt;
      }
      ADT_LET_CONST_REF(num_drr_op_operand_upstreams,
                        all_o_r_drr_graph.GetNumInputs(drr_op_operand));
      if (num_drr_op_operand_upstreams == 0) {
        return std::nullopt;
      }
      ADT_CHECK(num_drr_op_operand_upstreams == 1);
    }
    ADT_LET_CONST_REF(
        drr_op_operand_upstream,
        all_o_r_drr_graph.CastSoleUnignoredInput<DrrNativeIrOpResult>(
            drr_op_operand));
    ADT_LET_CONST_REF(
        pir_op_operand_upstream,
        topo_match_ctx->GetSoleBigGraphNode(drr_op_operand_upstream->node));
    ADT_LET_CONST_REF(pir_native_ir_value,
                      CastPirSoleOutput<PirNativeIrValue>(
                          default_pir_graph, pir_op_operand_upstream));
    adt::List<PirNativeIrOpOperand> pir_op_operands{};
    {
      auto DoEachDownstream =
          [&](const DrrGraphNode& node) -> adt::Result<adt::Ok> {
        ADT_LET_CONST_REF(drr_node, node.Get());
        ADT_CHECK(drr_node.Has<DrrNativeIrOpOperand>());
        ADT_LET_CONST_REF(pir_node, topo_match_ctx->GetSoleBigGraphNode(node));
        ADT_LET_CONST_REF(pir_native_ir_op_operand,
                          pir_node.TryGet<PirNativeIrOpOperand>());
        ADT_LET_CONST_REF(cur_pir_native_ir_value,
                          CastPirSoleInput<PirNativeIrValue>(
                              default_pir_graph, pir_native_ir_op_operand));
        if (cur_pir_native_ir_value == pir_native_ir_value) {
          pir_op_operands->push_back(pir_native_ir_op_operand);
        }
        return adt::Ok{};
      };
      ADT_RETURN_IF_ERR(all_o_r_drr_graph.VisitDownstreamNodes(
          drr_op_result->node, DoEachDownstream));
      if (pir_op_operands->empty()) {
        return std::nullopt;
      }
    }
    return RefNodeInfo{pir_native_ir_value, pir_op_operands};
  }

  adt::Result<std::optional<RefNodeInfo>> GetOutputRefNodeInfo(
      const GraphMatchCtx& graph_match_ctx,
      const DrrOptPackedIrOpResult& drr_op_result) const {
    DefaultGraph<DrrGraphNode> default_drr_graph{};
    DefaultGraph<PirNode> default_pir_graph{};
    AllOperandAndResultGraph<DrrGraphNode> all_o_r_drr_graph{};
    const auto& topo_match_ctx = graph_match_ctx->topo_match_ctx;
    ADT_LET_CONST_REF(
        drr_op_operand,
        all_o_r_drr_graph.CastSoleUnignoredInput<DrrOptPackedIrOpOperand>(
            drr_op_result));
    {
      ADT_LET_CONST_REF(num_drr_op_result_downstreams,
                        all_o_r_drr_graph.GetNumOutputs(drr_op_result));
      if (num_drr_op_result_downstreams != 0) {
        return std::nullopt;
      }
      ADT_LET_CONST_REF(num_drr_op_operand_upstreams,
                        all_o_r_drr_graph.GetNumInputs(drr_op_operand));
      if (num_drr_op_operand_upstreams == 0) {
        return std::nullopt;
      }
      ADT_CHECK(num_drr_op_operand_upstreams == 1);
    }
    ADT_LET_CONST_REF(
        drr_op_operand_upstream,
        all_o_r_drr_graph.CastSoleUnignoredInput<DrrNativeIrOpResult>(
            drr_op_operand));
    ADT_LET_CONST_REF(
        pir_op_operand_upstream,
        topo_match_ctx->GetSoleBigGraphNode(drr_op_operand_upstream->node));
    ADT_LET_CONST_REF(pir_native_ir_value,
                      CastPirSoleOutput<PirNativeIrValue>(
                          default_pir_graph, pir_op_operand_upstream));
    ADT_LET_CONST_REF(
        drr_ir_value,
        default_drr_graph.CastSoleUnignoredInput<DrrNativeIrValue>(
            drr_op_operand));
    std::unordered_set<PirNativeIrOpOperand> excluded;
    {
      auto DoEachDownstream =
          [&](const DrrGraphNode& node) -> adt::Result<adt::Ok> {
        ADT_LET_CONST_REF(drr_node, node.Get());
        if (!drr_node.Has<DrrNativeIrOpOperand>()) {
          return adt::Ok{};
        }
        ADT_LET_CONST_REF(pir_node, topo_match_ctx->GetSoleBigGraphNode(node));
        ADT_LET_CONST_REF(pir_op_operand,
                          pir_node.TryGet<PirNativeIrOpOperand>());
        ADT_CHECK(excluded.emplace(pir_op_operand).second);
        return adt::Ok{};
      };
      ADT_RETURN_IF_ERR(default_drr_graph.VisitDownstreamNodes(
          drr_ir_value->node, DoEachDownstream));
    }
    adt::List<PirNativeIrOpOperand> pir_op_operands{};
    {
      auto DoEachDownstream = [&](const PirNode& node) -> adt::Result<adt::Ok> {
        if (!node.Has<PirNativeIrOpOperand>()) {
          return adt::Ok{};
        }
        ADT_LET_CONST_REF(pir_op_operand, node.TryGet<PirNativeIrOpOperand>());
        if (excluded.count(pir_op_operand) == 0) {
          pir_op_operands->push_back(pir_op_operand);
        }
        return adt::Ok{};
      };
      ADT_RETURN_IF_ERR(default_pir_graph.VisitDownstreamNodes(
          pir_native_ir_value, DoEachDownstream));
    }
    return RefNodeInfo{pir_native_ir_value, pir_op_operands};
  }

  adt::Result<std::optional<RefNodeInfo>> GetInputRefNodeInfo(
      const GraphMatchCtx& graph_match_ctx,
      const DrrOptPackedIrOpResult& drr_op_result) const {
    DefaultGraph<PirNode> default_pir_graph{};
    AllOperandAndResultGraph<DrrGraphNode> all_o_r_drr_graph{};
    const auto& topo_match_ctx = graph_match_ctx->topo_match_ctx;
    ADT_LET_CONST_REF(
        drr_op_operand,
        all_o_r_drr_graph.CastSoleUnignoredInput<DrrOptPackedIrOpOperand>(
            drr_op_result));
    {
      ADT_LET_CONST_REF(num_drr_op_result_downstreams,
                        all_o_r_drr_graph.GetNumOutputs(drr_op_result));
      if (num_drr_op_result_downstreams == 0) {
        return std::nullopt;
      }
      ADT_LET_CONST_REF(num_drr_op_operand_upstreams,
                        all_o_r_drr_graph.GetNumInputs(drr_op_operand));
      if (num_drr_op_operand_upstreams != 0) {
        return std::nullopt;
      }
    }
    std::optional<PirNativeIrValue> pir_native_ir_value;
    adt::List<PirNativeIrOpOperand> pir_op_operands{};
    {
      auto DoEachDownstream =
          [&](const DrrGraphNode& node) -> adt::Result<adt::Ok> {
        ADT_LET_CONST_REF(drr_node, node.Get());
        ADT_CHECK(drr_node.Has<DrrNativeIrOpOperand>());
        ADT_LET_CONST_REF(pir_node, topo_match_ctx->GetSoleBigGraphNode(node));
        ADT_LET_CONST_REF(pir_native_ir_op_operand,
                          pir_node.TryGet<PirNativeIrOpOperand>());
        ADT_LET_CONST_REF(cur_pir_native_ir_value,
                          CastPirSoleInput<PirNativeIrValue>(
                              default_pir_graph, pir_native_ir_op_operand));
        if (!pir_native_ir_value.has_value()) {
          ADT_LET_CONST_REF(
              cur_pir_native_ir_value_upstream,
              GetPirSoleInput(default_pir_graph, cur_pir_native_ir_value));
          if (!cur_pir_native_ir_value_upstream.Has<PirNativeIrOpResult>()) {
            return adt::Ok{};
          }
          pir_native_ir_value = cur_pir_native_ir_value;
        }
        ADT_CHECK(cur_pir_native_ir_value == pir_native_ir_value.value());
        pir_op_operands->push_back(pir_native_ir_op_operand);
        return adt::Ok{};
      };
      ADT_RETURN_IF_ERR(all_o_r_drr_graph.VisitDownstreamNodes(
          drr_op_result->node, DoEachDownstream));
    }
    if (!pir_native_ir_value.has_value()) {
      return std::nullopt;
    }
    if (pir_op_operands->empty()) {
      return std::nullopt;
    }
    return RefNodeInfo{pir_native_ir_value.value(), pir_op_operands};
  }

  template <typename PirNodeImplT, typename GraphT>
  adt::Result<PirNodeImplT> CastPirSoleOutput(const GraphT& pir_graph,
                                              const PirNode& node) const {
    std::optional<PirNodeImplT> opt_pir_node{};
    auto DoEachDownstream =
        [&](const PirNode& downstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(pir_node_impl, downstream.TryGet<PirNodeImplT>());
      ADT_CHECK(!opt_pir_node.has_value());
      opt_pir_node = pir_node_impl;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(pir_graph.VisitDownstreamNodes(node, DoEachDownstream));
    ADT_CHECK(opt_pir_node.has_value());
    return opt_pir_node.value();
  }

  template <typename PirNodeImplT, typename GraphT>
  adt::Result<PirNodeImplT> CastPirSoleInput(const GraphT& pir_graph,
                                             const PirNode& node) const {
    std::optional<PirNodeImplT> opt_pir_node{};
    auto DoEachUpstream = [&](const PirNode& upstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(pir_node_impl, upstream.TryGet<PirNodeImplT>());
      ADT_CHECK(!opt_pir_node.has_value());
      opt_pir_node = pir_node_impl;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(pir_graph.VisitUpstreamNodes(node, DoEachUpstream));
    ADT_CHECK(opt_pir_node.has_value());
    return opt_pir_node.value();
  }

  template <typename GraphT>
  adt::Result<PirNode> GetPirSoleInput(const GraphT& pir_graph,
                                       const PirNode& node) const {
    std::optional<PirNode> opt_pir_node{};
    auto DoEachUpstream = [&](const PirNode& upstream) -> adt::Result<adt::Ok> {
      ADT_CHECK(!opt_pir_node.has_value());
      opt_pir_node = upstream;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(pir_graph.VisitUpstreamNodes(node, DoEachUpstream));
    ADT_CHECK(opt_pir_node.has_value());
    return opt_pir_node.value();
  }

  template <typename GraphT>
  adt::Result<std::size_t> GetNumPirOutputs(const GraphT& pir_graph,
                                            const PirNode& node) const {
    std::size_t num_outputs = 0;
    auto DoEachDownstream =
        [&](const PirNode& downstream) -> adt::Result<adt::Ok> {
      ++num_outputs;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(pir_graph.VisitDownstreamNodes(node, DoEachDownstream));
    return num_outputs;
  }

  template <typename GraphT>
  adt::Result<std::size_t> GetNumPirInputs(const GraphT& pir_graph,
                                           const PirNode& node) const {
    std::size_t num_inputs = 0;
    auto DoEachUpstream = [&](const PirNode& upstream) -> adt::Result<adt::Ok> {
      ++num_inputs;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(pir_graph.VisitUpstreamNodes(node, DoEachUpstream));
    return num_inputs;
  }

  adt::Result<DrrGraphNode> GetFirstNativeDrrIrOpResult(
      const DrrNativeIrOp& op) const {
    ADT_LET_CONST_REF(downstreams, op->node.DownstreamNodes());
    ADT_CHECK(downstreams.size() > 0);
    using List = adt::List<DrrGraphNode>;
    using Vec = ap::graph::IndexedTag<List>;
    ADT_LET_CONST_REF(indexed_list, downstreams.template TryGet<Vec>());
    return indexed_list.data->at(0);
  }

  adt::Result<PirNode> GetFirstNativePirIrOpResult(pir::Operation* op) const {
    ADT_CHECK(!op->isa<cinn::dialect::FusionOp>());
    ADT_CHECK(op->num_results() > 0);
    pir::Value value = op->result(0);
    ap::paddle::NativeIrOpResult ir_op_result{
        pir::OpResult::dyn_cast_from(value)};
    return ir_op_result;
  }
};

class DefaultAnchorApLowerFusionOpPattern : public pir::RewritePattern {
 private:
  ApLowerFusionOpPatternCtx ctx_;
  ApRewriter ap_rewriter_;
  mutable std::unordered_set<pir::Operation*> rewrited_;

 public:
  DefaultAnchorApLowerFusionOpPattern(pir::IrContext* ir_context,
                                      const ApLowerFusionOpPatternCtx& ctx)
      : pir::RewritePattern(ctx.anchor_op_name, 1, ir_context, {}),
        ctx_(ctx),
        ap_rewriter_(ctx) {}

  bool MatchAndRewrite(
      pir::Operation* op,
      pir::PatternRewriter& rewriter) const override {  // // NOLINT
    if (rewrited_.count(op) > 0) {
      return false;
    }
    const auto& ret = TryMatchAndRewrite(op, &rewriter);
    if (ret.HasError()) {
      LOG(ERROR) << "\nTraceback (most recent call last):\n"
                 << ret.GetError().CallStackToString() << "\n"
                 << ret.GetError().class_name() << ": " << ret.GetError().msg();
      return false;
    }
    LOG(ERROR) << "MatchAndRewrite: op: " << op
               << ", ret: " << ret.GetOkValue();
    rewrited_.insert(op);
    return ret.GetOkValue();
  }

  adt::Result<bool> TryMatchAndRewrite(pir::Operation* op,
                                       pir::PatternRewriter* rewriter) const {
    ADT_LET_CONST_REF(match_ctx, GetMatchCtx(op));
    ADT_CHECK(ctx_.drr_ctx->pass_name.has_value());
    LOG(ERROR) << "drr: " << ctx_.drr_ctx->pass_name.value() << " matched.";
    return ap_rewriter_.Rewrite(match_ctx, op, rewriter);
  }

  adt::Result<GraphMatchCtx> GetMatchCtx(pir::Operation* op) const {
    auto* parent_block = op->GetParent();
    ADT_CHECK(parent_block != nullptr);
    auto* parent_op = parent_block->GetParentOp();
    ADT_CHECK(!parent_op->isa<cinn::dialect::FusionOp>());
    const auto& default_anchor = ctx_.default_anchor;
    using Default = ap::drr::topo_kind::Default;
    ap::graph::GraphDescriptor<PirNode, Default> pir_graph{};
    ap::graph::GraphDescriptor<DrrGraphNode, Default> src_ptn_graph{};
    ap::ir_match::GraphMatcher<PirNode, Default, Default> graph_matcher(
        pir_graph, src_ptn_graph);
    ADT_LET_CONST_REF(
        anchor_cstr,
        src_ptn_graph.GetSmallGraphNodeCstr(default_anchor.node()));
    const auto& obj_node = CastToPirNode(op);
    ADT_LET_CONST_REF(satisfy_constraint,
                      pir_graph.Satisfy(obj_node, anchor_cstr));
    ADT_CHECK(satisfy_constraint) << adt::errors::ValueError{
        "pir_graph.Satisfy(obj_node, anchor_cstr) test failed."};
    ADT_LET_CONST_REF(
        graph_ctx,
        graph_matcher.MatchByAnchor(obj_node, default_anchor.node()));
    ADT_LET_CONST_REF(
        graph_matched,
        graph_matcher.IsGraphMatched(graph_ctx, default_anchor.node()));
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
      if (pattern_ctx.native_op_anchor.has_value()) {
        ps->Add(std::make_unique<NativeOpAnchorApLowerFusionOpPattern>(
            context, pattern_ctx));
      } else {
        ps->Add(std::make_unique<DefaultAnchorApLowerFusionOpPattern>(
            context, pattern_ctx));
      }
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitEachDrrCtx(AddFusionOpPattern));
    return adt::Ok{};
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitEachDrrCtx(const DoEachT& DoEach) {
    ADT_RETURN_IF_ERR(VisitEachDrrCtxByDrrPassRegistryItems(DoEach));
    return adt::Ok{};
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitEachDrrCtxByDrrPassRegistryItems(
      const DoEachT& DoEach) {
    ADT_LET_CONST_REF(registry, ApRegistryHelper{}.SingltonRegistry());
    const auto& drr_pass_registry_items = registry->drr_pass_registry_items;
    for (const auto& [drr_pass_name, nice2drr_pass_items] :
         drr_pass_registry_items) {
      std::optional<DrrCtx> opt_drr_ctx;
      for (const auto& [nice, drr_pass_items] : nice2drr_pass_items) {
        if (opt_drr_ctx.has_value()) {
          break;
        }
        for (const auto& drr_pass_item : drr_pass_items) {
          const auto& drr_ctx = GetDrrCtx(drr_pass_item);
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

  adt::Result<DrrCtx> GetDrrCtx(
      const ap::registry::DrrPassRegistryItem& drr_pass_item) {
    ADT_LET_CONST_REF(drr_ctx, ApDrrHelper{}.Interpret(drr_pass_item));
    if (!drr_ctx->pass_name.has_value()) {
      drr_ctx.shared_ptr()->pass_name = drr_pass_item->drr_pass_name;
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
