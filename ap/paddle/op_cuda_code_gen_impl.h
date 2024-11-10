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

#include <sstream>
#include "ap/axpr/anf_expr_util.h"
#include "ap/axpr/cps_expr_interpreter.h"
#include "ap/axpr/data_type_util.h"
#include "ap/axpr/lambda_expr_builder.h"
#include "ap/axpr/pointer_type_util.h"
#include "ap/code_gen/code_gen_ctx.h"
#include "ap/code_gen/dim_expr_kernel_arg_id.h"
#include "ap/code_gen/op_code_gen_ctx.h"
#include "ap/code_gen/op_cuda_gen_impl.h"
#include "ap/drr/node.h"
#include "ap/drr/topo_kind.h"
#include "ap/drr/value.h"
#include "ap/graph/node.h"
#include "ap/index_expr/index_tuple_expr_cuda_code_generator.h"
#include "ap/ir_match/native_or_ref_ir_value.h"
#include "ap/op_compute/value.h"
#include "ap/op_compute/value_method_class.h"
#include "ap/paddle/indexed_ir_graph_util.h"
#include "ap/paddle/pir_graph_descriptor.h"
#include "ap/paddle/pir_node.h"
#include "ap/registry/registry.h"
#include "ap/registry/registry_mgr.h"
#include "ap/registry/registry_singleton.h"
#include "ap/registry/value.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/pir/include/core/builtin_type.h"

namespace ap::paddle {

struct OpCudaCodeGenImpl {
  using BirNode = PirNode;
  using OpCodeGenCtx = code_gen::OpCodeGenCtx<BirNode>;
  using IrOp = code_gen::IrOp<BirNode>;

  using LocalVarBinding = code_gen::LocalVarBinding<BirNode>;
  using LocalVarBindingList = std::vector<LocalVarBinding>;

  using DrrValue = drr::Value;
  using DrrNode = drr::Node<DrrValue>;
  using DrrGraphNode = graph::Node<DrrNode>;
  using DrrPackedIrOp = drr::PackedIrOp<DrrValue, DrrNode>;
  using DrrOptPackedIrOp = drr::OptPackedIrOp<DrrValue, DrrNode>;
  using DrrOptPackedIrOpOperand = drr::OptPackedIrOpOperand<DrrNode>;
  using DrrOptPackedIrOpResult = drr::OptPackedIrOpResult<DrrNode>;

  using DrrTrivialFusionIrOpImpl =
      std::variant<DrrPackedIrOp, DrrOptPackedIrOp>;
  struct DrrTrivialFusionIrOp : public DrrTrivialFusionIrOpImpl {
    using DrrTrivialFusionIrOpImpl::DrrTrivialFusionIrOpImpl;
    DEFINE_ADT_VARIANT_METHODS(DrrTrivialFusionIrOpImpl);

    DrrGraphNode node() const {
      return Match([](const auto& impl) { return impl->node; });
    }
  };

  using DrrNativeIrValue = drr::NativeIrValue<DrrNode>;
  using DrrPackedIrValue = drr::PackedIrValue<DrrNode>;
  using IndexTupleExpr = index_expr::IndexTupleExpr;

  using GraphMatchCtx = ir_match::GraphMatchCtx<BirNode>;

  using IndexTupleExprCodeGenerator =
      index_expr::IndexTupleExprCudaCodeGenerator;

  using Registry = registry::Registry;

  adt::Result<std::string> CodeGen(const OpCodeGenCtx& op_code_gen_ctx,
                                   const IrOp& ir_op) {
    using RetString = adt::Result<std::string>;
    return ir_op.Match(
        [&](const PackedIrOp& packed_ir_op) -> RetString {
          return PackedIrOpCodeGen(op_code_gen_ctx, packed_ir_op);
        },
        [&](const RefIrOp& ref_ir_op) -> RetString {
          return RefIrOpCodeGen(op_code_gen_ctx, ref_ir_op);
        },
        [&](const auto&) -> RetString {
          return adt::errors::TypeError{
              std::string() +
              "paddle pir only support generating code for packed ir op."};
        });
  }

  using DefaultDrrGraph =
      graph::GraphDescriptor<DrrGraphNode, drr::topo_kind::Default>;
  adt::Result<std::string> RefIrOpCodeGen(const OpCodeGenCtx& op_code_gen_ctx,
                                          const RefIrOp& ref_ir_op) {
    ADT_LET_CONST_REF(in_out, GetInputOutput(op_code_gen_ctx, ref_ir_op));
    const auto& [input, output] = in_out;
    ADT_LET_CONST_REF(input_var_name,
                      GetBoundLocalVarName(op_code_gen_ctx, input));
    ADT_LET_CONST_REF(output_var_name,
                      GetBoundLocalVarName(op_code_gen_ctx, output));
    return std::string() + output_var_name + " = " + input_var_name + ";\n";
  }

  using NativeOrRefIrValue = ir_match::NativeOrRefIrValue<BirNode>;

  adt::Result<std::string> GetBoundLocalVarName(
      const OpCodeGenCtx& op_code_gen_ctx, const NativeOrRefIrValue& ir_value) {
    for (const auto& [var_name, v] : op_code_gen_ctx->local_var_binding) {
      if (v == ir_value) {
        return var_name;
      }
    }
    return adt::errors::KeyError{"no local var bounded."};
  }

  adt::Result<std::pair<NativeIrValue, RefIrValue>> GetInputOutput(
      const OpCodeGenCtx& op_code_gen_ctx, const RefIrOp& ref_ir_op) const {
    DefaultDrrGraph drr_graph;
    ADT_LET_CONST_REF(graph_match_ctx, GetGraphMatchCtx(op_code_gen_ctx));
    ADT_LET_CONST_REF(drr_opt_packed_ir_op_node,
                      graph_match_ctx->GetMatchedSmallGraphNode(ref_ir_op));
    ADT_LET_CONST_REF(drr_opt_packed_ir_op, drr_opt_packed_ir_op_node.Get());
    ADT_LET_CONST_REF(
        drr_opt_op_operand,
        drr_graph.template CastSoleUnignoredInput<DrrOptPackedIrOpOperand>(
            drr_opt_packed_ir_op));
    ADT_LET_CONST_REF(
        drr_op_input,
        drr_graph.template CastSoleUnignoredInput<DrrNativeIrValue>(
            drr_opt_op_operand));
    ADT_LET_CONST_REF(
        drr_opt_op_result,
        drr_graph.template CastSoleUnignoredOutput<DrrOptPackedIrOpResult>(
            drr_opt_packed_ir_op));
    ADT_LET_CONST_REF(
        drr_op_output,
        drr_graph.template CastSoleUnignoredOutput<DrrNativeIrValue>(
            drr_opt_op_result));
    ADT_LET_CONST_REF(pir_op_input,
                      graph_match_ctx->GetSoleBigGraphNode(drr_op_input->node));
    ADT_LET_CONST_REF(
        pir_op_output,
        graph_match_ctx->GetSoleBigGraphNode(drr_op_output->node));
    ADT_LET_CONST_REF(pir_op_input_ir_value,
                      pir_op_input.template TryGet<NativeIrValue>());
    ADT_LET_CONST_REF(pir_op_output_ir_ref_value,
                      pir_op_output.template TryGet<RefIrValue>());
    return std::make_pair(pir_op_input_ir_value, pir_op_output_ir_ref_value);
  }

  adt::Result<std::string> PackedIrOpCodeGen(
      const OpCodeGenCtx& op_code_gen_ctx, const PackedIrOp& packed_ir_op) {
    const auto& loop_indexes_expr = op_code_gen_ctx->loop_index_tuple_expr;
    ADT_LET_CONST_REF(
        ir_graph,
        CreatePureElementwiseIndexedIrGraph(packed_ir_op, loop_indexes_expr));
    ADT_LET_CONST_REF(ss,
                      IrGraphCodeGen(op_code_gen_ctx, ir_graph, packed_ir_op));
    return ss.str();
  }

  adt::Result<std::ostringstream> IrGraphCodeGen(
      const OpCodeGenCtx& op_code_gen_ctx,
      const IndexedIrGraph& ir_graph,
      const PackedIrOp& packed_ir_op) {
    return ir_graph.Match(
        [&](const auto& impl) -> adt::Result<std::ostringstream> {
          return IrGraphCodeGenImpl(op_code_gen_ctx, impl, packed_ir_op);
        });
  }

  using IrGraphNode = graph::Node<IndexedIrNode>;

  struct IrGraphNodeInfoImpl {
    std::optional<std::string> global_ptr_name;
    std::string local_var_name;
  };
  DEFINE_ADT_RC(IrGraphNodeInfo, IrGraphNodeInfoImpl);

  struct IrGraphTranslateCtx {
    std::unordered_map<IrGraphNode, IrGraphNodeInfo> node2value;

    adt::Result<adt::Ok> Emplace(const IrGraphNode& node,
                                 const IrGraphNodeInfo& value) {
      ADT_CHECK(TryEmplace(node, value));
      return adt::Ok{};
    }

    bool TryEmplace(const IrGraphNode& node, const IrGraphNodeInfo& value) {
      return this->node2value.emplace(node, value).second;
    }

    adt::Result<IrGraphNodeInfo> Get(const IrGraphNode& node) const {
      const auto& iter = this->node2value.find(node);
      ADT_CHECK(iter != this->node2value.end());
      return iter->second;
    }
  };

  adt::Result<std::ostringstream> IrGraphCodeGenImpl(
      const OpCodeGenCtx& op_code_gen_ctx,
      const PureElementwiseIndexedIrGraph& ir_graph,
      const PackedIrOp& packed_ir_op) {
    std::ostringstream ss;
    IrGraphTranslateCtx ctx{};
    const auto& loop_var_names = op_code_gen_ctx->loop_var_names;
    using OptStr = std::optional<std::string>;
    auto ArgName4DimExpr = [&](const symbol::DimExpr& dim_expr) -> OptStr {
      code_gen::DimExprKernelArgId<PirNode> kernel_arg_id{dim_expr};
      const auto iter =
          op_code_gen_ctx->kernel_arg_id2arg_name.find(kernel_arg_id);
      if (iter == op_code_gen_ctx->kernel_arg_id2arg_name.end()) {
        return std::nullopt;
      }
      return iter->second;
    };
    IndexTupleExprCodeGenerator indexes_expr_gen(
        &ss, loop_var_names, ArgName4DimExpr);
    ADT_RETURN_IF_ERR(CodeGenInputs(
        &ss, &ctx, &indexes_expr_gen, op_code_gen_ctx, ir_graph, packed_ir_op));
    ADT_RETURN_IF_ERR(CodeGenBody(&ss, &ctx, op_code_gen_ctx, ir_graph));
    ADT_RETURN_IF_ERR(CodeGenOutputs(
        &ss, &ctx, &indexes_expr_gen, op_code_gen_ctx, ir_graph, packed_ir_op));
    return ss;
  }

  adt::Result<adt::Ok> CodeGenInputs(
      std::ostringstream* ss,
      IrGraphTranslateCtx* ctx,
      IndexTupleExprCodeGenerator* indexes_expr_gen,
      const OpCodeGenCtx& op_code_gen_ctx,
      const PureElementwiseIndexedIrGraph& ir_graph,
      const PackedIrOp& packed_ir_op) {
    LOG(ERROR) << "CodeGenInputs enter:\n" << ss->str();
    std::unordered_set<pir::Value> registered_values;
    auto DoEachDeclare = [&](pir::Value value) -> adt::Result<adt::Ok> {
      if (!registered_values.emplace(value).second) {
        return adt::Ok{};
      }
      ADT_LET_CONST_REF(node_info,
                        MakeInputIrGraphNodeInfo(op_code_gen_ctx, value));
      ADT_LET_CONST_REF(ir_value, ir_graph->GetIndexedIrValue(value));
      LOG(ERROR) << "CodeGenInputs value: " << value.impl() << ", node_id: "
                 << std::to_string(ir_value->node.node_id().value());
      ADT_RETURN_IF_ERR(
          InitIrGraphTranslateCtxNodeInfo(ctx, ir_value, node_info));
      ADT_RETURN_IF_ERR(GenLoadCode(ss, indexes_expr_gen, ir_value, node_info));
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitInputBirNativeIrValue(
        op_code_gen_ctx, packed_ir_op, DoEachDeclare));
    LOG(ERROR) << "CodeGenInputs leave:\n" << ss->str();
    return adt::Ok{};
  }

  adt::Result<adt::Ok> GenLoadCode(
      std::ostringstream* ss,
      IndexTupleExprCodeGenerator* indexes_expr_gen,
      const IndexedIrValue<IndexedIrNode>& ir_value,
      const IrGraphNodeInfo& node_info) {
    if (!node_info->global_ptr_name.has_value()) {
      return adt::Ok{};
    }
    const auto& global_ptr_name = node_info->global_ptr_name.value();
    const auto& indexes_expr = ir_value->indexes_expr;
    ADT_LET_CONST_REF(index_var_name, indexes_expr_gen->CodeGen(indexes_expr));
    (*ss) << "auto " << node_info->local_var_name << " = " << global_ptr_name
          << "[" << index_var_name << "];\n";
    return adt::Ok{};
  }

  adt::Result<IrGraphNodeInfo> MakeInputIrGraphNodeInfo(
      const OpCodeGenCtx& op_code_gen_ctx, pir::Value value) {
    using InArg = code_gen::InTensorDataPtrKernelArgId<BirNode>;
    return MakeIrGraphNodeInfo<InArg>(op_code_gen_ctx, value);
  }

  adt::Result<IrGraphNodeInfo> MakeOutputIrGraphNodeInfo(
      const OpCodeGenCtx& op_code_gen_ctx, pir::Value value) {
    using OutArg = code_gen::OutTensorDataPtrKernelArgId<BirNode>;
    return MakeIrGraphNodeInfo<OutArg>(op_code_gen_ctx, value);
  }

  template <typename KernelArgIdImpl>
  adt::Result<IrGraphNodeInfo> MakeIrGraphNodeInfo(
      const OpCodeGenCtx& op_code_gen_ctx, pir::Value value) {
    ADT_LET_CONST_REF(
        opt_kernel_arg_name,
        GetKernelArgName<KernelArgIdImpl>(op_code_gen_ctx, value));
    std::optional<std::string> default_global_ptr_name;
    std::optional<std::string> default_local_var_name;
    if (opt_kernel_arg_name.has_value()) {
      default_global_ptr_name = opt_kernel_arg_name.value();
      default_local_var_name = opt_kernel_arg_name.value() + "_local_var";
    }
    return MakeIrGraphNodeInfoByNames(op_code_gen_ctx,
                                      default_global_ptr_name,
                                      default_local_var_name,
                                      value);
  }

  template <typename KernelArgIdImpl>
  adt::Result<std::optional<std::string>> GetKernelArgName(
      const OpCodeGenCtx& op_code_gen_ctx, pir::Value value) {
    BirNode pir_node{NativeIrValue{value}};
    KernelArgIdImpl kernel_arg_id_impl{pir_node};
    code_gen::KernelArgId<BirNode> kernel_arg_id{kernel_arg_id_impl};
    const auto& iter =
        op_code_gen_ctx->kernel_arg_id2arg_name.find(kernel_arg_id);
    if (iter == op_code_gen_ctx->kernel_arg_id2arg_name.end()) {
      return std::nullopt;
    }
    return iter->second;
  }

  adt::Result<IrGraphNodeInfo> MakeIrGraphNodeInfoByNames(
      const OpCodeGenCtx& op_code_gen_ctx,
      const std::optional<std::string>& default_global_ptr_name,
      const std::optional<std::string>& default_local_var_name,
      pir::Value value) {
    ADT_LET_CONST_REF(opt_local_var_name,
                      GetReplacedLocalVar(op_code_gen_ctx, value));
    std::optional<std::string> global_ptr_name{};
    std::string local_var_name{};
    if (opt_local_var_name.has_value()) {
      global_ptr_name = std::nullopt;
      local_var_name = opt_local_var_name.value();
    } else {
      ADT_CHECK(default_global_ptr_name.has_value());
      global_ptr_name = default_global_ptr_name.value();
      ADT_CHECK(default_local_var_name.has_value());
      local_var_name = default_local_var_name.value();
    }
    return IrGraphNodeInfo{global_ptr_name, local_var_name};
  }

  adt::Result<adt::Ok> CodeGenOutputs(
      std::ostringstream* ss,
      IrGraphTranslateCtx* ctx,
      IndexTupleExprCodeGenerator* indexes_expr_gen,
      const OpCodeGenCtx& op_code_gen_ctx,
      const PureElementwiseIndexedIrGraph& ir_graph,
      const PackedIrOp& packed_ir_op) {
    std::unordered_set<pir::Value> registered_values;
    auto DoEachDeclare = [&](pir::Value value) -> adt::Result<adt::Ok> {
      if (!registered_values.emplace(value).second) {
        return adt::Ok{};
      }
      ADT_LET_CONST_REF(yield_op_input,
                        GetYieldOpInputIndexedIrValue(ir_graph, value));
      ADT_LET_CONST_REF(yield_op_input_node_info,
                        ctx->Get(yield_op_input->node));
      ADT_LET_CONST_REF(output_node_info,
                        MakeOutputIrGraphNodeInfo(op_code_gen_ctx, value));
      ADT_RETURN_IF_ERR(GenStoreCode(ss,
                                     indexes_expr_gen,
                                     yield_op_input->indexes_expr,
                                     yield_op_input_node_info,
                                     output_node_info));
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(
        VisitOutputNativeIrValue(op_code_gen_ctx, packed_ir_op, DoEachDeclare));
    return adt::Ok{};
  }

  adt::Result<IndexedIrValue<IndexedIrNode>> GetYieldOpInputIndexedIrValue(
      const PureElementwiseIndexedIrGraph& ir_graph, pir::Value output) {
    ADT_CHECK(ir_graph->yield_op_inputs.size() == ir_graph->outputs.size());
    std::optional<IndexedIrValue<IndexedIrNode>> yield_op_input;
    for (int i = 0; i < ir_graph->yield_op_inputs.size(); ++i) {
      if (ir_graph->outputs.at(i) == output) {
        yield_op_input = ir_graph->yield_op_inputs.at(i);
        break;
      }
    }
    ADT_CHECK(yield_op_input.has_value())
        << adt::errors::KeyError{std::string() + "no yield_op_input found."};
    return yield_op_input.value();
  }

  adt::Result<adt::Ok> GenStoreCode(
      std::ostringstream* ss,
      IndexTupleExprCodeGenerator* indexes_expr_gen,
      const index_expr::IndexTupleExpr& indexes_expr,
      const IrGraphNodeInfo& yield_op_input_node_info,
      const IrGraphNodeInfo& output_node_info) {
    if (!output_node_info->global_ptr_name.has_value()) {
      (*ss) << output_node_info->local_var_name << " = "
            << yield_op_input_node_info->local_var_name << ";\n";
    } else {
      ADT_LET_CONST_REF(index_var_name,
                        indexes_expr_gen->CodeGen(indexes_expr));
      const auto& global_ptr_name = output_node_info->global_ptr_name.value();
      (*ss) << global_ptr_name << "[" << index_var_name
            << "] = " << yield_op_input_node_info->local_var_name << ";\n";
    }
    return adt::Ok{};
  }

  adt::Result<std::string> IndexesExprCodeGen(
      const IndexTupleExpr& indexes_expr, std::ostringstream* ss) {}

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitInputBirNativeIrValue(
      const OpCodeGenCtx& op_code_gen_ctx,
      const PackedIrOp& packed_ir_op,
      const DoEachT& DoEach) {
    ADT_LET_CONST_REF(graph_match_ctx, GetGraphMatchCtx(op_code_gen_ctx));
    ADT_LET_CONST_REF(drr_trivial_fusion_ir_op,
                      GetDrrTrivialFusionIrOp(graph_match_ctx, packed_ir_op));
    auto DoEachNativeValue =
        [&](const auto& drr_ir_value) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(value, GetPirValue(graph_match_ctx, drr_ir_value));
      return DoEach(value);
    };
    auto DoEachPackedValue =
        [&](const auto& drr_ir_value) -> adt::Result<adt::Ok> {
      ADT_RETURN_IF_ERR(
          VisitPackedPirValue(graph_match_ctx, drr_ir_value, DoEach));
      return adt::Ok{};
    };
    return VisitDrrTrivialFusionIrOpInput(
        drr_trivial_fusion_ir_op, DoEachNativeValue, DoEachPackedValue);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitPackedPirValue(const GraphMatchCtx& matc_ctx,
                                           const DrrPackedIrValue& drr_ir_value,
                                           const DoEachT& DoEach) {
    auto DoEachPirNode = [&](const PirNode& pir_node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(pir_value, pir_node.template TryGet<NativeIrValue>());
      ADT_RETURN_IF_ERR(DoEach(pir_value.value));
      return adt::Ok{};
    };
    const auto& node = drr_ir_value->node;
    ADT_RETURN_IF_ERR(
        matc_ctx->VisitPackedBigGraphIrValueNode(node, DoEachPirNode));
    return adt::Ok{};
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitOutputNativeIrValue(
      const OpCodeGenCtx& op_code_gen_ctx,
      const PackedIrOp& packed_ir_op,
      const DoEachT& DoEach) {
    ADT_LET_CONST_REF(graph_match_ctx, GetGraphMatchCtx(op_code_gen_ctx));
    ADT_LET_CONST_REF(drr_trivial_fusion_ir_op,
                      GetDrrTrivialFusionIrOp(graph_match_ctx, packed_ir_op));
    auto DoEachNativeValue =
        [&](const auto& drr_ir_value) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(value, GetPirValue(graph_match_ctx, drr_ir_value));
      return DoEach(value);
    };
    auto DoEachPackedValue =
        [&](const auto& drr_ir_value) -> adt::Result<adt::Ok> {
      ADT_RETURN_IF_ERR(
          VisitPackedPirValue(graph_match_ctx, drr_ir_value, DoEach));
      return adt::Ok{};
    };
    return VisitDrrTrivialFusionIrOpOutput(
        drr_trivial_fusion_ir_op, DoEachNativeValue, DoEachPackedValue);
  }

  template <typename DoEachNativeValueT, typename DoEachPackedValueT>
  adt::Result<adt::Ok> VisitDrrTrivialFusionIrOpInput(
      const DrrTrivialFusionIrOp& drr_trivial_fusion_ir_op,
      const DoEachNativeValueT& DoEachNativeValue,
      const DoEachPackedValueT DoEachPackedValue) {
    LOG(ERROR) << "drr_trivial_fusion_ir_op: "
               << graph::NodeDescriptor<DrrGraphNode>{}.DebugId(
                      drr_trivial_fusion_ir_op.node());
    auto DoEach = [&](const DrrGraphNode& node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(drr_node, node.Get());
      LOG(ERROR) << "drr_trivial_fusion_ir_op input: "
                 << graph::NodeDescriptor<DrrGraphNode>{}.DebugId(node);
      return drr_node.Match(
          [&](const DrrNativeIrValue& ir_value) -> adt::Result<adt::Ok> {
            return DoEachNativeValue(ir_value);
          },
          [&](const DrrPackedIrValue& ir_value) -> adt::Result<adt::Ok> {
            return DoEachPackedValue(ir_value);
          },
          [&](const auto&) -> adt::Result<adt::Ok> {
            return adt::errors::ValueError{
                "the second connected upstreams of drr packed ir op should be "
                "drr native ir values or drr packed ir values."};
          });
    };
    return VisitSecondConnectedUpstream(drr_trivial_fusion_ir_op.node(),
                                        DoEach);
  }

  template <typename DoEachNativeValueT, typename DoEachPackedValueT>
  adt::Result<adt::Ok> VisitDrrTrivialFusionIrOpOutput(
      const DrrTrivialFusionIrOp& drr_trivial_fusion_ir_op,
      const DoEachNativeValueT& DoEachNativeValue,
      const DoEachPackedValueT DoEachPackedValue) {
    auto DoEach = [&](const DrrGraphNode& node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(drr_node, node.Get());
      return drr_node.Match(
          [&](const DrrNativeIrValue& ir_value) -> adt::Result<adt::Ok> {
            return DoEachNativeValue(ir_value);
          },
          [&](const DrrPackedIrValue& ir_value) -> adt::Result<adt::Ok> {
            return DoEachPackedValue(ir_value);
          },
          [&](const auto&) -> adt::Result<adt::Ok> {
            return adt::errors::ValueError{
                "the second connected upstreams of drr packed ir op should be "
                "drr native ir values or drr packed ir values."};
          });
    };
    return VisitSecondConnectedDownstream(drr_trivial_fusion_ir_op.node(),
                                          DoEach);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitSecondConnectedUpstream(const DrrGraphNode& node,
                                                    const DoEachT& DoEach) {
    auto DoEachUpstream = [&](const auto& upstream) -> adt::Result<adt::Ok> {
      return VisitUpstream(upstream, DoEach);
    };
    return VisitUpstream(node, DoEachUpstream);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitSecondConnectedDownstream(const DrrGraphNode& node,
                                                      const DoEachT& DoEach) {
    auto DoEachUpstream = [&](const auto& downstream) -> adt::Result<adt::Ok> {
      return VisitDownstream(downstream, DoEach);
    };
    return VisitDownstream(node, DoEachUpstream);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitUpstream(const DrrGraphNode& node,
                                     const DoEachT& DoEach) {
    ADT_LET_CONST_REF(upstreams, node.UpstreamNodes());
    return upstreams.VisitNodes(DoEach);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitDownstream(const DrrGraphNode& node,
                                       const DoEachT& DoEach) {
    ADT_LET_CONST_REF(downstreams, node.DownstreamNodes());
    return downstreams.VisitNodes(DoEach);
  }

  adt::Result<axpr::Lambda<axpr::CoreExpr>> GetNativeIrValueGetterLambda(
      const DrrNativeIrValue& drr_native_ir_value) {
    ADT_LET_CONST_REF(anf_expr,
                      GetNativeIrValueGetterAnfExpr(drr_native_ir_value));
    const auto& core_expr = axpr::ConvertAnfExprToCoreExpr(anf_expr);
    ADT_LET_CONST_REF(
        atomic, core_expr.template TryGet<axpr::Atomic<axpr::CoreExpr>>());
    ADT_LET_CONST_REF(lambda,
                      atomic.template TryGet<axpr::Lambda<axpr::CoreExpr>>());
    return lambda;
  }

  adt::Result<axpr::AnfExpr> GetNativeIrValueGetterAnfExpr(
      const DrrNativeIrValue& drr_native_ir_value) {
    axpr::LambdaExprBuilder lmd{};
    const std::string& name = drr_native_ir_value->name;
    return lmd.Lambda({"ctx"}, [&](auto& ctx) {
      return ctx.Var("ctx").Attr("tensor").Attr(name).Attr("data_ptr");
    });
  }

  adt::Result<pir::Value> GetPirValue(
      const GraphMatchCtx& graph_match_ctx,
      const DrrNativeIrValue& drr_native_ir_value) {
    const auto& node = drr_native_ir_value->node;
    ADT_LET_CONST_REF(pir_node, graph_match_ctx->GetSoleBigGraphNode(node));
    ADT_LET_CONST_REF(pir_value, pir_node.template TryGet<NativeIrValue>());
    return pir_value.value;
  }

  adt::Result<DrrTrivialFusionIrOp> GetDrrTrivialFusionIrOp(
      const GraphMatchCtx& graph_match_ctx, const PackedIrOp& packed_ir_op) {
    ADT_LET_CONST_REF(node,
                      graph_match_ctx->GetMatchedSmallGraphNode(packed_ir_op));
    ADT_LET_CONST_REF(drr_node, node.Get());
    using RetT = adt::Result<DrrTrivialFusionIrOp>;
    return drr_node.Match(
        [&](const DrrPackedIrOp& impl) -> RetT { return impl; },
        [&](const DrrOptPackedIrOp& impl) -> RetT { return impl; },
        [&](const auto&) -> RetT {
          return adt::errors::NotImplementedError{
              "convertion from DrrNode to DrrTrivialFusionIrOp failed."};
        });
  }

  adt::Result<GraphMatchCtx> GetGraphMatchCtx(
      const OpCodeGenCtx& op_code_gen_ctx) const {
    ADT_LET_CONST_REF(code_gen_ctx,
                      adt::WeakPtrLock(op_code_gen_ctx->code_gen_ctx));
    ADT_CHECK(code_gen_ctx->ir_match_ctx.has_value());
    const auto& ir_match_ctx = code_gen_ctx->ir_match_ctx.value();
    return ir_match_ctx->graph_match_ctx;
  }

  adt::Result<axpr::PointerType> GetConstDataPointerType(pir::Value value) {
    ADT_LET_CONST_REF(data_type, ConvertToDataType(value));
    return axpr::GetConstPointerTypeFromDataType(data_type);
  }

  adt::Result<axpr::PointerType> GetMutableDataPointerType(pir::Value value) {
    ADT_LET_CONST_REF(data_type, ConvertToDataType(value));
    return axpr::GetMutablePointerTypeFromDataType(data_type);
  }

  adt::Result<axpr::DataType> ConvertToDataType(pir::Value value) {
    ADT_LET_CONST_REF(dtype, ConvertToPhiDataType(value));
    return ap::axpr::GetDataTypeFromPhiDataType(dtype);
  }

  adt::Result<phi::DataType> ConvertToPhiDataType(pir::Value value) {
    ADT_LET_CONST_REF(type, GetPirDataType(value));
    try {
      return ::paddle::dialect::TransToPhiDataType(type);
    } catch (const std::exception& e) {
      return adt::errors::TypeError{
          "failed to cast from pir data type to phi data type."};
    }
  }

  adt::Result<pir::Type> GetPirDataType(pir::Value value) {
    if (!value.type().isa<pir::DenseTensorType>()) {
      return adt::errors::NotImplementedError{
          "pir value must be of DenseTensorType"};
    }
    const auto dense_tensor_type =
        value.type().dyn_cast<pir::DenseTensorType>();
    return dense_tensor_type.dtype();
  }

  adt::Result<adt::Ok> InitIrGraphTranslateCtxNodeInfo(
      IrGraphTranslateCtx* ctx,
      const IndexedIrValue<IndexedIrNode>& ir_value,
      const IrGraphNodeInfo& node_info) {
    ADT_RETURN_IF_ERR(ctx->Emplace(ir_value->node, node_info));
    return adt::Ok{};
  }

  adt::Result<std::optional<std::string>> GetReplacedLocalVar(
      const OpCodeGenCtx& op_code_gen_ctx, pir::Value value) {
    ADT_LET_CONST_REF(local_var_bindings, GetLocalVarBindings(op_code_gen_ctx));
    for (const auto& [local_var_name, native_or_ref] : *local_var_bindings) {
      ADT_LET_CONST_REF(native_ir_value, native_or_ref.TryGet<NativeIrValue>());
      if (native_ir_value.value == value) {
        return std::optional<std::string>{local_var_name};
      }
    }
    return std::optional<std::string>{std::nullopt};
  }

  adt::Result<const LocalVarBindingList*> GetLocalVarBindings(
      const OpCodeGenCtx& op_code_gen_ctx) {
    return &op_code_gen_ctx->local_var_binding;
  }

  adt::Result<adt::Ok> CodeGenBody(
      std::ostringstream* ss,
      IrGraphTranslateCtx* ctx,
      const OpCodeGenCtx& op_code_gen_ctx,
      const PureElementwiseIndexedIrGraph& ir_graph) {
    auto DoEach =
        [&](const IndexedIrOp<IndexedIrNode>& ir_op) -> adt::Result<adt::Ok> {
      return CodeGenOpCompute(ss, ctx, op_code_gen_ctx, ir_op);
    };
    return VisitOrderedIndexedIrOp(ir_graph, DoEach);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitOrderedIndexedIrOp(
      const PureElementwiseIndexedIrGraph& ir_graph, const DoEachT& DoEach) {
    for (const auto& node : ir_graph->node_arena->nodes()) {
      if (node.template Has<IndexedIrOp<IndexedIrNode>>()) {
        ADT_RETURN_IF_ERR(
            DoEach(node.template Get<IndexedIrOp<IndexedIrNode>>()));
      }
    }
    return adt::Ok{};
  }

  adt::Result<adt::Ok> CodeGenOpCompute(
      std::ostringstream* ss,
      IrGraphTranslateCtx* ctx,
      const OpCodeGenCtx& op_code_gen_ctx,
      const IndexedIrOp<IndexedIrNode>& ir_op) {
    ADT_LET_CONST_REF(lambda, GetOpComputeLambda(op_code_gen_ctx, ir_op));
    ADT_RETURN_IF_ERR(InsertOutputVarNameToCtx(ctx, ir_op));
    ADT_LET_CONST_REF(input_var_names, GetInputVarNames(*ctx, ir_op));
    ADT_LET_CONST_REF(output_var_names, GetOutputVarNames(*ctx, ir_op));
    ADT_LET_CONST_REF(output_type_names, GetOutputTypeNames(ir_op));
    ADT_RETURN_IF_ERR(
        CodeGenOutputLocalVarDeclares(ss, output_type_names, output_var_names));
    return CodeGenOpComputeByLambda(
        ss, lambda, ir_op, input_var_names, output_var_names);
  }

  adt::Result<axpr::Lambda<axpr::CoreExpr>> GetOpComputeLambda(
      const OpCodeGenCtx& op_code_gen_ctx,
      const IndexedIrOp<IndexedIrNode>& ir_op) {
    const auto& op_name = ir_op->op->name();
    std::optional<axpr::Lambda<axpr::CoreExpr>> ret;
    auto FetchLambda = [&](const auto& cell) -> adt::Result<adt::LoopCtrl> {
      ret = cell->data;
      if (ret.has_value()) {
        return adt::Break{};
      } else {
        return adt::Continue{};
      }
    };
    ADT_RETURN_IF_ERR(ForEachOpComputeLambda(op_name, FetchLambda));
    ADT_CHECK(ret.has_value()) << adt::errors::AttributeError{
        std::string() + "no op_compute lambda registered for op name '" +
        op_name + "'."};
    return ret.value();
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> ForEachOpComputeLambda(const std::string& op_name,
                                              const DoEachT& DoEach) {
    ADT_RETURN_IF_ERR(ap::registry::RegistryMgr::Singleton()->LoadAllOnce());
    using RegistryVal = ap::registry::Value;
    ADT_LET_CONST_REF(registry, ap::registry::RegistrySingleton::Singleton());
    const auto& key2nice2op_computes = registry->op_compute_registry_items;
    const auto& iter = key2nice2op_computes.find(op_name);
    ADT_CHECK(iter != key2nice2op_computes.end())
        << adt::errors::AttributeError{
               std::string() + "no op_compute lambda registered for op name '" +
               op_name + "'."};
    const auto& nice2op_computes = iter->second;
    for (const auto& [nice, op_computes] : nice2op_computes) {
      for (const auto& op_compute : op_computes) {
        if (op_compute->arch_type == "cuda") {
          ADT_LET_CONST_REF(loop_ctrl, DoEach(op_compute->lambda));
          if (loop_ctrl.template Has<adt::Break>()) {
            break;
          }
        }
      }
    }
    return adt::Ok{};
  }

  adt::Result<std::vector<std::string>> GetOutputTypeNames(
      const IndexedIrOp<IndexedIrNode>& ir_op) {
    ADT_LET_CONST_REF(downstreams, ir_op->node.DownstreamNodes());
    std::vector<std::string> ret{};
    ret.reserve(downstreams.size());
    auto DoEach = [&](const auto& node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(ir_node, node.Get());
      ADT_LET_CONST_REF(
          ir_value, ir_node.template TryGet<IndexedIrValue<IndexedIrNode>>());
      ADT_LET_CONST_REF(data_type, ConvertToDataType(ir_value->value));
      ret.push_back(data_type.Name());
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(downstreams.VisitNodes(DoEach));
    return ret;
  }

  adt::Result<std::vector<std::string>> GetInputVarNames(
      const IrGraphTranslateCtx& ctx, const IndexedIrOp<IndexedIrNode>& ir_op) {
    ADT_LET_CONST_REF(upstreams, ir_op->node.UpstreamNodes());
    std::vector<std::string> ret{};
    ret.reserve(upstreams.size());
    auto DoEach = [&](const auto& node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(ir_node, node.Get());
      ADT_CHECK(ir_node.template Has<IndexedIrValue<IndexedIrNode>>());
      ADT_LET_CONST_REF(node_info, ctx.Get(node));
      ret.push_back(node_info->local_var_name);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(upstreams.VisitNodes(DoEach));
    return ret;
  }

  adt::Result<std::vector<std::string>> GetOutputVarNames(
      const IrGraphTranslateCtx& ctx, const IndexedIrOp<IndexedIrNode>& ir_op) {
    ADT_LET_CONST_REF(downstreams, ir_op->node.DownstreamNodes());
    std::vector<std::string> ret{};
    ret.reserve(downstreams.size());
    auto DoEach = [&](const auto& node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(ir_node, node.Get());
      ADT_CHECK(ir_node.template Has<IndexedIrValue<IndexedIrNode>>());
      ADT_LET_CONST_REF(node_info, ctx.Get(node));
      ret.push_back(node_info->local_var_name);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(downstreams.VisitNodes(DoEach));
    return ret;
  }

  adt::Result<adt::Ok> InsertOutputVarNameToCtx(
      IrGraphTranslateCtx* ctx, const IndexedIrOp<IndexedIrNode>& ir_op) {
    ADT_LET_CONST_REF(downstreams, ir_op->node.DownstreamNodes());
    auto DoEach = [&](const auto& node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(ir_node, node.Get());
      ADT_CHECK(ir_node.template Has<IndexedIrValue<IndexedIrNode>>());
      IrGraphNodeInfo node_info{std::nullopt,
                                ap::common::NewUniqueId("_ap_local_var")};
      ADT_RETURN_IF_ERR(ctx->Emplace(node, node_info));
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(downstreams.VisitNodes(DoEach));
    return adt::Ok{};
  }

  adt::Result<adt::Ok> CodeGenOutputLocalVarDeclares(
      std::ostringstream* ss,
      const std::vector<std::string>& output_type_names,
      const std::vector<std::string>& output_var_names) {
    ADT_CHECK(output_var_names.size() == output_type_names.size());
    for (int i = 0; i < output_var_names.size(); ++i) {
      (*ss) << output_type_names.at(i) << " " << output_var_names.at(i)
            << ";\n";
    }
    return adt::Ok{};
  }

  adt::Result<adt::Ok> CodeGenOpComputeByLambda(
      std::ostringstream* ss,
      const axpr::Lambda<axpr::CoreExpr>& lambda,
      const IndexedIrOp<IndexedIrNode>& ir_op,
      const std::vector<std::string>& input_var_names,
      const std::vector<std::string>& output_var_names) {
    const auto& inputs = ConvertToOpComputeList(input_var_names);
    op_compute::Val inputs_val{inputs};
    const auto& outputs = ConvertToOpComputeList(output_var_names);
    op_compute::Val outputs_val{outputs};
    // TODO(tianchao): support op attributes.
    const auto& attrs = axpr::Object<op_compute::Val>{};
    op_compute::Val attrs_val{attrs};
    axpr::CpsExprInterpreter<op_compute::Val> cps_expr_interpreter;
    ADT_LET_CONST_REF(op_compute_code_gen_ret,
                      cps_expr_interpreter.Interpret(
                          lambda, {inputs_val, outputs_val, attrs_val}));
    ADT_LET_CONST_REF(op_compute_str,
                      op_compute_code_gen_ret.template TryGet<std::string>());
    (*ss) << op_compute_str << ";\n";
    return adt::Ok{};
  }

  adt::List<op_compute::Value> ConvertToOpComputeList(
      const std::vector<std::string>& var_names) {
    adt::List<op_compute::Value> ret{};
    ret->reserve(var_names.size());
    for (const auto& var_name : var_names) {
      ret->emplace_back(var_name);
    }
    return ret;
  }
};

}  // namespace ap::paddle

namespace ap::code_gen {

template <>
struct OpCudaCodeGenImpl<ap::paddle::PirNode>
    : public paddle::OpCudaCodeGenImpl {};

}  // namespace ap::code_gen
