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

#include "paddle/cinn/hlir/dialect/operator/ir/fusion_descriptor.h"
#include <functional>
#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/collect_sym_expr.h"
#include "paddle/common/bfs_walker.h"
#include "paddle/common/enforce.h"
#include "paddle/common/flags.h"
#include "paddle/common/topo_walker.h"
#include "paddle/pir/include/dialect/pexpr/index_expr_interpreter.h"
#include "paddle/pir/include/dialect/pexpr/op_index_tuple_expr_signature.h"

COMMON_DECLARE_bool(enable_debug_ap);

namespace ap {

std::optional<OpOutArg> OpOutArg::MakeFromValue(pir::Value value) {
  auto* owner = value.owner();
  if (owner == nullptr) {
    return std::nullopt;
  }
  for (int i = 0; i < owner->num_results(); ++i) {
    if (owner->result(i) == value) {
      return OpOutArg{value, i};
    }
  }
  return std::nullopt;
}

using DimExprs4ValueT =
    std::function<std::optional<const symbol::ShapeOrDataDimExprs*>(
        pir::Value)>;
using Nice2IndexLambdas4OpT =
    std::function<std::optional<Nice2IndexLambdas>(const pir::Operation*)>;
using IndexClosure4OpT =
    std::function<std::optional<pexpr::IndexClosure>(const pir::Operation*)>;

using OpOrArgImpl = std::variant<OpInArg, OpOutArg, pir::Operation*>;

struct OpOrArg : public OpOrArgImpl {
  using OpOrArgImpl::OpOrArgImpl;
  DEFINE_ADT_VARIANT_METHODS(OpOrArgImpl);

  size_t GetHashValue() const {
    return Match([](const auto& impl) {
      return std::hash<std::decay_t<decltype(impl)>>()(impl);
    });
  }
};

}  // namespace ap

namespace std {

template <>
struct hash<ap::OpOrArg> {
  size_t operator()(const ap::OpOrArg& op_or_arg) const {
    return op_or_arg.GetHashValue();
  }
};

}  // namespace std

namespace ap {

namespace {

using pexpr::OpIndexTupleExprSignature;

bool Op2Anchor2IndexesExprSignatureImpl::IsReachable(const OpOrArg& src,
                                                     const OpOrArg& dst) const {
  return std::visit(
      ::common::Overloaded{
          [&](const OpInArg& src_op_operand, const pir::Operation* dst_op) {
            // src_op_operand must be an anchorable OpArg.
            const auto& iter = op2sigatures.find(dst_op);
            if (iter == op2sigatures.end()) {
              return false;
            }
            const auto& args = iter->second;
            return args.in_arg2signature.count(src) > 0;
          },
          [&](const pir::Operation* src_op, const OpInArg& dst_op_operand) {
            // src_op must have at least one anchorable OpArg.
            const auto& iter = op2sigatures.find(src_op);
            if (iter == op2sigatures.end()) {
              return false;
            }
            const auto& args = iter->second;
            if (args.in_arg2signature.empty() &&
                args.out_arg2signature.empty()) {
              return false;
            }
            return dst_op_operand.owner() == src_op;
          },
          [&](const OpOutArg& src_op_result, const pir::Operation* dst_op) {
            // src_op_result must be an anchorable OpArg.
            const auto& iter = op2sigatures.find(dst_op);
            if (iter == op2sigatures.end()) {
              return false;
            }
            const auto& args = iter->second;
            return args.out_arg2signature.count(src) > 0;
          },
          [&](const pir::Operation* src_op, const OpOutArg& dst_op_result) {
            // src_op must have at least one anchorable OpArg.
            const auto& iter = op2sigatures.find(src_op);
            if (iter == op2sigatures.end()) {
              return false;
            }
            const auto& args = iter->second;
            if (args.in_arg2signature.empty() &&
                args.out_arg2signature.empty()) {
              return false;
            }
            return dst_op_result.owner() == src_op;
          },
          [](const auto&, const auto&) { return false; }},
      src.variant(),
      dst.variant);
}

::common::TopoWalker<OpOrArg> GetAnchorableTopoWalker(
    const ::common::TopoWalker<OpOrArg>& walker,
    const Op2Anchor2IndexesExprSignature& op2anchor2indexes_expr_signature) {
  const auto ForEachUpstream = [=](const OpOrArg& op_or_arg,
                                   const auto& DoEach) {
    walker.VisitPrevNodes(op_or_arg, [&](const OpOrArg& upstream) {
      if (op2anchor2indexes_expr_signature.IsReachable(op_or_arg, upstream)) {
        DoEach(upstream);
      }
    });
  };
  const auto ForEachDownstream = [=](const OpOrArg& op_or_arg,
                                     const auto& DoEach) {
    walker.VisitNextNodes(op_or_arg, [&](const OpOrArg& downstream) {
      if (op2anchor2indexes_expr_signature.IsReachable(op_or_arg, downstream)) {
        DoEach(downstream);
      }
    });
  };
  return ::common::TopoWalker<OpOrArg>(ForEachUpstream, ForEachDownstream);
}

::common::TopoWalker<OpOrArg> GetOpOrArgTopoWalker(pir::Block* block) {
  auto ops = std::make_shared<std::unordered_set<pir::Operation*>>();
  for (auto& op : *block) {
    ops->insert(&op);
  }
  const auto ForEachUpstream = [ops](const OpOrArg& op_or_arg,
                                     const auto& DoEach) {
    op_or_arg.Match(
        [&](OpInArg op_operand) {
          auto* op = op_operand.source().owner();
          if (ops->count(op) == 0) {
            return;
          }
          const auto& out_arg = OpOutArg::MakeFromValue(op_operand.source());
          if (out_arg.has_value()) {
            DoEach(OpOrArg{out_arg.value()});
          }
        },
        [&](OpOutArg value) { DoEach(OpOrArg{value.owner()}); },
        [&](pir::Operation* op) {
          for (int i = 0; i < op->num_operands(); ++i) {
            DoEach(OpOrArg{OpInArg{op->operand(i)}});
          }
        });
  };
  const auto ForEachDownstream = [ops](const OpOrArg& op_or_arg,
                                       const auto& DoEach) {
    op_or_arg.Match(
        [&](OpInArg op_operand) {
          auto* op = op_operand.owner();
          DoEach(OpOrArg{op});
        },
        [&](OpOutArg value) {
          for (iter = value.use_begin(); iter != value.use_end(); ++iter) {
            if (ops->count(iter->owner()) == 0) {
              continue;
            }
            DoEach(OpOrArg{OpInArg{*iter}});
          }
        },
        [&](pir::Operation* op) {
          for (int i = 0; i < op->num_results(); ++i) {
            DoEach(OpOrArg{OpOutArg{op->result(i), i}});
          }
        });
  };
  return ::common::TopoWalker<OpOrArg>(ForEachUpstream, ForEachDownstream);
}

std::optional<pir::Operation*> GetYieldOp(
    const cinn::dialect::FusionOp& fusion_op) {
  for (const auto& op : *fusion_op.block()) {
    if (op.isa<pir::YieldOp>()) {
      return &op;
    }
  }
  return std::nullopt;
}

const ::common::BfsWalker<OpOrArg> GetBfsWalker(
    const ::common::TopoWalker<OpOrArg>& walker) {
  const auto ForEachNext = [walker](const OpOrArg& node, const auto& DoEach) {
    walker.VisitPrevNodes(node, DoEach);
    walker.VisitNextNodes(node, DoEach);
  };
  return ::common::BfsWalker<OpOrArg>(ForEachNext);
}

void CheckReachableToAllOps(const std::string& error_tip,
                            const ::common::TopoWalker<OpOrArg>& walker,
                            const cinn::dialect::FusionOp& fusion_op) {
  const auto& bfs_walker = GetBfsWalker(walker);
  const auto yield_op = GetYieldOp(fusion_op);
  PADDLE_ENFORCE_EQ(yield_op.has_value(),
                    true,
                    phi::errors::InvalidArgument(
                        "no yield op found in cinn::dialect::FusionOp"));
  OpOrArg start{yield_op.value()};
  std::unordered_set<pir::Operation*> visited_ops;
  bfs_walker(start, [&](const OpOrArg& node) {
    node.Match([&](pir::Operation* op) { visited_ops.insert(op); },
               [&](const auto&) {
                 // Do nothing.
               });
  });
  for (const auto& op : *fusion_op.block()) {
    if (op.isa<pir::YieldOp>()) {
      continue;
    }
    PADDLE_ENFORCE_GT(
        visited_ops.count(&op),
        0 phi::errors::Unimplemented(
            "%s: op named `%s' is not visisted.", error_tip, op.name()));
  }
}

class OpArg2OpIndexesExprSignatureInferrer final {
 public:
  OpArg2OpIndexesExprSignatureInferrer(
      const cinn::dialect::FusionOp& fusion_op,
      const DimExprs4ValueT& GetDimExprs4Value,
      const IndexClosure4OpT& GetIndexClosure4Op)
      : fusion_op_(fusion_op),
        interpreter_(),
        DimExprs4Value(GetDimExprs4Value),
        IndexClosure4Op(GetIndexClosure4Op) {}

  Op2Anchor2IndexesExprSignature GetOp2Anchor2IndexesExprSignature() {
    std::unordered_map<pir::Operation*, OpArg2OpIndexesExprSignature>
        op2sigatures;
    for (const auto& op : *fusion_op.block()) {
      const auto& op_indexes_expr_signature =
          GetOpArg2OpIndexesExprSignature(&op);
      if (!op_indexes_expr_signature.has_value()) {
        continue;
      }
      op2sigatures[&op] = op_indexes_expr_signature.value();
    }
    return Op2Anchor2IndexesExprSignature{std::move(op2sigatures)};
  }

 private:
  std::optional<OpArg2OpIndexesExprSignature> GetOpArg2OpIndexesExprSignature(
      const pir::Operation* op) {
    OpArg2OpIndexesExprSignature ret;
    const index_closure = IndexClosure4Op(op);
    if (!index_closure.has_value()) {
      return std::nullopt;
    }
    VisitEachInArg(op, [&](const OpInArg& op_in_arg, const auto& shape) {
      if (const auto& indexex_expr = CalIndexTupleExpr(index_closure, shape)) {
        (*ret.in_arg2signature)[OpArg{op_in_arg}] = indexex_expr.value();
      }
    });
    VisitEachOutArg(op, [&](const OpOutArg& op_out_arg, const auto& shape) {
      if (const auto& indexex_expr = CalIndexTupleExpr(index_closure, shape)) {
        (*ret.out_arg2signature)[OpArg{op_out_arg}] = indexex_expr.value();
      }
    });
    return ret;
  }

  template <typename DoEachT>
  void VisitEachInArg(const pir::Operation* op, const DoEachT& DoEach) {
    for (int i = 0; i < op->num_operands(); ++i) {
      OpInArg in_arg(op->operand(i));
      DoEach(in_arg, GetShape(op->operand_source(i)));
    }
  }

  template <typename DoEachT>
  void VisitEachOutArg(const pir::Operation* op, const DoEachT& DoEach) {
    for (int i = 0; i < op->num_results(); ++i) {
      OpOutArg out_arg(op->result(i));
      DoEach(out_arg, GetShape(op->result(i)));
    }
  }

  std::optional<std::List<symbol::DimExpr>> GetShape(pir::Value value) {
    const auto& shape_or_data = DimExprs4Value(value);
    if (!shape_or_data.has_value()) {
      return std::nullopt;
    }
    using OptDimList = std::optional<std::List<symbol::DimExpr>>;
    return shape_or_data.value()->Match(
        [&](const symbol::TensorShapeOrDataDimExprs& impl) -> OptDimList {
          adt::List<symbol::DimExpr> dim_exprs;
          dim_exprs.reserve(vec.size());
          for (const auto& dim_expr : vec) {
            dim_exprs.emplace_back(dim_expr);
          }
          return dim_exprs;
        },
        [&](const auto&) -> OptDimList { return std::nullopt; });
  }

  std::optional<OpIndexTupleExprSignature> CalIndexTupleExpr(
      const IndexClosure& index_closure,
      const std::optional<adt::List<symbol::DimExpr>>& shape) {
    if (!shape.has_value()) {
      return std::nullopt;
    }
    const auto& res =
        index_closure(interpreter_, IndexTupleExprDomain{shape.value()});
    if (!res.Has<OpIndexTupleExprSignature>()) {
      return std::nullopt;
    }
    return res.Get<OpIndexTupleExprSignature>();
  }

  cinn::dialect::FusionOp fusion_op_;
  pexpr::IndexExprInterpreter interpreter_;
  DimExprs4ValueT DimExprs4Value;
  IndexClosure4OpT IndexClosure4Op;
};

pexpr::IndexClosure MakeIndexClosure(
    const std::shared_ptr<IndexExprInterpreter>& interpreter,
    const OpIndexClosureData& closure_data) {
  const auto& func = [interpreter,
                      closure_data](const IndexTupleExpr& indexes_expr)
      -> adt::Result<OpIndexTupleExprSignature> {
    std::vector<pexpr::Val> args{closure_data.ctx,
                                 closure_data.inputs_meta,
                                 closure_data.outputs_meta,
                                 closure_data.in_vars};
    const auto& res = interpreter();
  };
  return pexpr::NativeIndexClosure(func);
}

pexpr::index_expr::Val MakeTensorShapeFromVec(
    const std::vector<symbol::DimExpr>& vec) {
  using pexpr::index_expr::Val;
  adt::List<Val> dim_exprs;
  dim_exprs.reserve(vec.size());
  for (const auto& dim_expr : vec) {
    dim_exprs.emplace_back(Val{pexpr::index_expr::IndexExprValue{dim_expr}});
  }
  return dim_exprs;
}

pexpr::index_expr::Val MakeTensorShapeImpl(
    const symbol::TensorShapeOrDataDimExprs& impl) {
  return MakeTensorShapeFromVec(impl.shape());
}

pexpr::index_expr::Val MakeTensorDataImpl(
    const symbol::TensorShapeOrDataDimExprs& impl) {
  if (!impl.data().has_value()) {
    return pexpr::index_expr::Val{adt::Nothing{}};
  }
  return MakeTensorShapeFromVec(impl.data().value());
}

pexpr::index_expr::Val MakeTensorShapeImpl(
    const symbol::TensorListShapeOrDataDimExprs& impl) {
  using pexpr::index_expr::Val;
  adt::List<Val> dim_exprs;
  dim_exprs.reserve(impl.size());
  for (const auto& shape_or_data : impl) {
    dim_exprs.emplace_back(MakeTensorShapeImpl(shape_or_data));
  }
  return Val{dim_exprs};
}

pexpr::index_expr::Val MakeTensorDataImpl(
    const symbol::TensorListShapeOrDataDimExprs& impl) {
  using pexpr::index_expr::Val;
  adt::List<Val> dim_exprs;
  dim_exprs.reserve(impl.size());
  for (const auto& shape_or_data : impl) {
    dim_exprs.emplace_back(MakeTensorDataImpl(shape_or_data));
  }
  return Val{dim_exprs};
}

pexpr::index_expr::Val MakeTensorShape(
    const symbol::ShapeOrDataDimExprs& shape_or_data) {
  using pexpr::index_expr::Val;
  return shape_or_data.Match(
      [&](const symbol::NullShapeOrDataDimExpr& impl) {
        return Val{adt::Nothing{}};
      },
      [&](const symbol::TensorShapeOrDataDimExprs& impl) {
        return MakeTensorShapeImpl(impl);
      },
      [&](const symbol::TensorListShapeOrDataDimExprs& impl) {
        return MakeTensorShapeImpl(impl);
      },
      [&](const symbol::RankedTensorArrayShapeOrDataDimExprs& impl) {
        return Val{adt::Nothing{}};
      });
}

pexpr::index_expr::Val MakeTensorData(
    const symbol::ShapeOrDataDimExprs& shape_or_data) {
  using pexpr::index_expr::Val;
  return shape_or_data.Match(
      [&](const symbol::NullShapeOrDataDimExpr& impl) {
        return Val{adt::Nothing{}};
      },
      [&](const symbol::TensorShapeOrDataDimExprs& impl) {
        return MakeTensorDataImpl(impl);
      },
      [&](const symbol::TensorListShapeOrDataDimExprs& impl) {
        return MakeTensorDataImpl(impl);
      },
      [&](const symbol::RankedTensorArrayShapeOrDataDimExprs& impl) {
        return Val{adt::Nothing{}};
      });
}

pexpr::Object<pexpr::index_expr::Val> MakeTensorMetaObject(
    const symbol::ShapeOrDataDimExprs& shape_or_data) {
  return pexpr::Object<pexpr::index_expr::Val>{
      std::unordered_map<std::string, pexpr::index_expr::Val>{
          {"shape", MakeTensorShape(shape_or_data)},
          {"data", MakeTensorData(shape_or_data)},
      }};
}

std::optional<IndexClosureData> MakeIndexClosureData(
    const DimExprs4ValueT& DimExprs4Value, const pir::Operation& op) {
  using ValueList = adt::List<pexpr::index_expr::Val>;
  using OptValueList = std::optional<ValueList>;
  const auto& inputs_meta = [&]() -> OptValueList {
    ValueList ret;
    ret.reserve(op.num_operands());
    for (int i = 0; i < op.num_operands(); ++i) {
      const auto& dim_exprs = DimExprs4Value(op.operand_source(i));
      if (!dim_exprs.has_value()) {
        return std::nullopt;
      }
      ret.emplace_back(MakeTensorMetaObject(dim_exprs));
    }
    return ret;
  }();
  if (!inputs_meta.has_value()) {
    return std::nullopt;
  }
  const auto& outputs_meta = [&]() -> OptValueList {
    ValueList ret;
    ret.reserve(op.num_results());
    for (int i = 0; i < op.num_results(); ++i) {
      const auto& dim_exprs = DimExprs4Value(op.result(i));
      if (!dim_exprs.has_value()) {
        return std::nullopt;
      }
      ret.emplace_back(MakeTensorMetaObject(dim_exprs));
    }
    return ret;
  }();
  if (!outputs_meta.has_value()) {
    return std::nullopt;
  }
  const auto& in_vars = [&] {
    adt::List<pexpr::Val> in_vars;
    in_vars->reserve(op->num_operands());
    for (int i = 0; i < op->num_operands(); ++i) {
      in_vars->push_back(pexpr::Val{std::string() + "op" +
                                    std::to_string(op.id()) + "_in" +
                                    std::to_string(i)});
    }
    return in_vars;
  }();
  return IndexClosureData{
      .ctx = pexpr::Val{adt::Nothing{}},
      .inputs_meta = inputs_meta.value(),
      .outputs_meta = outputs_meta.value(),
      .in_vars = in_vars,
  };
}

IndexClosure4OpT MakeIndexClosure4Op(
    const std::shared_ptr<IndexExprInterpreter>& interpreter,
    const DimExprs4ValueT& DimExprs4Value,
    const Nice2IndexLambdas4OpT& Nice2IndexLambdas4Op,
    const cinn::dialect::FusionOp& fusion_op) {
  using Op2IndexClosure =
      std::unordered_map<const pir::Operation*, pexpr::IndexClosure>;
  auto op2closure = std::make_shared<Op2IndexClosure>();
  for (const auto& op : *fusion_op.block()) {
    if (op.isa<pir::YieldOp>()) {
      continue;
    }
    const nice2index_lambdas = Nice2IndexLambdas4Op(&op);
    if (!nice2index_lambdas.has_value()) {
      continue;
    }
    const auto& closure_data = MakeIndexClosureData(DimExprs4Value, op);
    if (!closure_data.value()) {
      continue;
    }
    (*op2closure)[&op] = IndexClosure{OrderedOneofIndexClosure{
        .interpreter = interpreter,
        .closure_data = closure_data.value(),
        .nice2index_lambdas = nice2index_lambdas.value(),
    }};
  }
  return [op2closure](
             const pir::Operation* op) -> std::optional<pexpr::IndexClosure> {
    const auto& iter = op2closure->find(op);
    if (iter == op2closure->end()) {
      return std::nullopt;
    }
    return iter->second;
  };
}

DimExprs4ValueT MakeDimExpr4Value(const cinn::dialect::FusionOp& fusion_op,
                                  ShapeConstraintIRAnalysis* shape_analysis) {
  const std::vector<pir::Operation*> ops = [&] {
    std::vector<pir::Operation*> ops;
    ops.reserve(fusion_op.block()->size());
    for (auto& op : *fusion_op.block()) {
      ops.emplace_back(&op);
    }
    return ops;
  }();
  using Value2DimExprs =
      std::unordered_map<::pir::Value, symbol::ShapeOrDataDimExprs>;
  Value2DimExprs value2dim_expr =
      cinn::dialect::ir::details::CreateGroupShapeOrDataExprs(ops,
                                                              *shape_analysis);
  return [map = move(value2dim_expr)](pir::Value value) {
    const auto& iter = map.find(value);
    if (iter == map.end()) {
      return std::nullopt;
    }
    return &iter->second;
  };
}

using OpOrArgPath = std::vector<OpOrArg>;
struct OpOrArgPathsImpl {
  OpArgTo<OpArgTo<OpOrArgPath>> paths;
};
DEFINE_ADT_RC(OpOrArgPaths, OpOrArgPathsImpl);

// feasible path not shortest path.
adt::Result<OpArgTo<OpOrArgPath>> GetPathsToYieldOpInArgs(
    const OpInArg src,
    const ::common::TopoWalker<OpOrArg>& walker,
    const pir::Operation* yield_op) {
  ::common::BfsWalker<OpOrArg> bfs_walker = GetBfsWalker(walker);
  OpArgTo<OpOrArgPath> ret;
  ret[src] = std::vector<OpOrArg>{OpOrArg{src}};
  const auto GetShortestNearbyPath =
      [&](const OpOrArg& dst) -> std::optional<const OpOrArgPath*> {
    std::optional<const OpOrArgPath*> path{std::nullopt};
    bfs_walker.VisitNextNodes(dst, [&](const OpOrArg& nearby) {
      const auto& iter = ret.find(nearby);
      if (iter == ret.end()) {
        return;
      }
      const auto* cur_path = &iter->second;
      if (!path.has_value()) {
        path = cur_path;
      } else if (cur_path->size() < path.value()->size()) {
        path = cur_path;
      } else {
        // Do nothing.
      }
    });
    return path;
  };
  bfs_walker(OpOrArg{src}, [&](const OpOrArg& dst) {
    const auto& near_path = GetShortestNearbyPath(dst);
    if (!near_path.has_value()) {
      return;
    }
    ret[dst].reserve(near_path.value()->size() + 1);
    ret[dst] = *near_path.value();
    ret[dst].push_back(dst);
  });
  return std::move(ret);
}

adt::Result<OpOrArgPaths> GetPathsBetweenYieldOpInArgs(
    const ::common::TopoWalker<OpOrArg>& walker,
    const pir::Operation* yield_op) {
  OpOrArgPaths ret;
  for (int i = 0; i < yield_op->num_operands(); ++i) {
    OpInArg in_arg{yield_op->operand(i)};
    adt::Result<OpArgTo<OpOrArgPath>> op_arg2path =
        GetPathsToYieldOpInArgs(in_arg, walker, yield_op);
    if (op_arg2path.HasOkValue()) {
      ret->paths[in_arg] = op_arg2path.GetOkValue();
    }
  }
  return ret;
}

struct OpArgStep {
  OpArg src;
  pir::Operation* op;
  OpArg dst;
};

struct OpArgPath {
  OpArg src;
  std::vector<OpArgStep> op_steps;
  OpArg dst;
};
struct OpArgPathsImpl {
  OpArgTo<OpArgTo<OpArgPath>> paths;
};
DEFINE_ADT_RC(OpArgPaths, OpArgPathsImpl);

adt::Result<OpArg> ConvertToOpArg(const OpOrArg& op_or_arg) {
  return op_or_arg.Match(
      [&](pir::Operation*) -> adt::Result<OpArg> {
        return adt::errors::InvalidArgument{
            "`pir::Operation*' couldn't convert to OpArg."};
      },
      [&](const auto& impl) -> adt::Result<OpArg> { return impl; });
}

adt::Result<OpArgStep> ConvertToOpStep(const OpOrArg& opt_src,
                                       const OpOrArg& opt_op,
                                       const OpOrArg& opt_dst) {
  const auto& src_op_arg = ConvertToOpArg(opt_src);
  const auto& dst_op_arg = ConvertToOpArg(opt_dst);
  const auto& pattern_match = ::common::Overloaded{
      [](const OpArg& src,
         pir::Operation* op,
         const OpArg& dst) -> adt::Result<OpArgStep> {
        return OpArgStep{src, op, dst};
      },
      [](const auto&, const auto&, const auto&) -> adt::Result<OpArgStep> {
        return adt::errors::InvalidArgument{
            "Couldn't convert 3 consecutive OpOrArg into OpArgStep."};
      }};
  return std::visit(pattern_match,
                    src_op_arg.variant(),
                    opt_op.variant(),
                    dst_op_arg.variant());
}

adt::Result<OpArgPath> ConvertToOpArgPath(const OpOrArgPath& op_or_arg_path) {
  if (op_or_arg_path.size() == 1) {
    const auto& op_arg = ConvertToOpArg(op_or_arg_path.at(0));
    ADT_RETURN_IF_ERROR(op_arg);
    return OpArgPath{
        .src = op_arg,
        .op_steps = std::vector<OpArgStep>{},
        .dst = op_arg,
    };
  }
  if (op_or_arg_path.size() < 5) {
    return adt::errors::InvalidArgument{
        "op_or_arg_path.size() should be >= 5."};
  }
  if ((op_or_arg_path.size() - 2) % 3 == 0) {
    return adt::errors::InvalidArgument{
        "(op_or_arg_path.size() - 2) should be divided by 3."};
  }
  OpArgPath ret;
  const auto& src_op_arg = ConvertToOpArg(op_or_arg_path.at(0));
  ADT_RETURN_IF_ERROR(src_op_arg);
  ret.src = src_op_arg.GetOkValue();
  for (int i = 1; (i + 3) < op_or_arg_path.size() - 1; i += 3) {
    const auto& op_step = ConvertToOpStep(op_or_arg_path.at(i),
                                          op_or_arg_path.at(i + 1),
                                          op_or_arg_path.at(i + 2));
    ADT_RETURN_IF_ERROR(op_step);
    ret.op_steps.emplace_back(op_step.GetOkValue());
  }
  const auto& dst_op_arg =
      ConvertToOpArg(op_or_arg_path.at(op_or_arg_path.size()));
  ADT_RETURN_IF_ERROR(dst_op_arg);
  ret.dst = dst_op_arg.GetOkValue();
  return std::move(ret);
}

adt::Result<OpArgPaths> ConvertPathsToOpArgPaths(
    const OpOrArgPaths& op_or_arg_paths) {
  OpArgPaths op_arg_paths;
  for (const auto& [src_op_arg, dst2paths] : op_or_arg_paths->paths) {
    for (const auto& [dst_op_arg, op_or_arg_path] : dst2paths) {
      const auto& converted = ConvertToOpArgPath(op_or_arg_path);
      ADT_RETURN_IF_ERROR(converted);
      op_arg_paths->paths[src_op_arg][dst_op_arg] = converted.GetOkValue();
    }
  }
  return op_arg_paths;
}

template <typename DoEachT>
void VisitOpInArg(const pir::Operation* op, const DoEachT& DoEach) {
  for (int i = 0; i < op->num_operands(); ++i) {
    OpArg in_arg{op->operand(i)};
    DoEach(in_arg, i);
  }
}

template <typename T, typename ContainerT>
adt::Result<T> VecGet(const ContainerT& vec, int idx) {
  if (idx < 0) {
    return adt::errors::InvalidArgument{
        "negative index for vector is not allowed."};
  }
  if (idx >= vec->size()) {
    return adt::errors::InvalidArgument{"index for vector is out of range."};
  }
  return vec->at(idx);
}

adt::Result<IndexTupleExpr> GetIndexTupleExprFromSignature(
    const OpIndexTupleExprSignature& op_indexes_expr_signature,
    const OpArg& op_arg) {
  return op_arg.Match(
      [&](const OpInArg& in_arg) -> adt::Result<IndexTupleExpr> {
        return VecGet<IndexTupleExpr>(
            op_indexes_expr_signature.in_signature.descriptors, in_arg.index());
      },
      [&](const OpOutArg& out_arg) -> adt::Result<IndexTupleExpr> {
        return VecGet<IndexTupleExpr>(
            op_indexes_expr_signature.out_signature.descriptors, out_arg.index);
      });
}

const char* GetArgTypeString(const OpArg& op_arg) {
  return op_arg.Match([](const OpInArg&) { return "in"; },
                      [](const OpOutArg&) { return "out"; });
}

size_t GetArgIndex(const OpArg& op_arg) {
  return op_arg.Match([](const OpInArg& impl) { return impl.index(); },
                      [](const OpOutArg& impl) { return impl.index; });
}

adt::Result<IndexTupleExpr> GetIndexesExpr(
    const OpArgStep& op_step,
    const Op2Anchor2IndexesExprSignature& op2anchor2indexes_expr_signature) {
  const auto& iter = op2anchor2indexes_expr_signature.find(op_step.op);
  if (iter == op2anchor2indexes_expr_signature.end()) {
    return adt::errors::InvalidArgument{
        std::string() +
        "op index signature not found. op_name: " + op_step.op->name()};
  }
  const auto& anchor2signature = iter->second;
  const auto& anchor_iter = anchor2signature.find(op_step.src);
  if (anchor_iter == anchor2signature.end()) {
    return adt::errors::InvalidArgument{
        std::string() + "index signature of op `" + op_step.op->name() +
        "' could not be infered from " + GetArgTypeString(op_step.src) +
        " arg " + std::to_string(GetArgIndex(op_step.src))};
  }
  const auto& signature = anchor_iter->second;
  return GetIndexTupleExprFromSignature(signature, op_step.dst);
}

adt::Result<TrackedIndexesTransform> MakeIndexesTransformByPath(
    const OpArgPath& op_arg_path,
    const Op2Anchor2IndexesExprSignature& op2anchor2indexes_expr_signature) {
  if (op_arg_path.op_steps.empty()) {
    return TrackedIndexesTransform{adt::IdentityFunc{}};
  }
  const auto& opt_indexes_expr = GetIndexesExpr(
      op_arg_path.op_steps.at(0), op2anchor2indexes_expr_signature);
  ADT_RETURN_IF_ERROR(opt_indexes_expr);
  IndexTupleExpr indexes_expr = opt_indexes_expr.GetOkValue();
  ValidIndexExprBuilder builder{};
  for (int i = 1; i < op_arg_path.op_steps.size(); ++i) {
    const auto& op_step = op_arg_path.op_steps.at(i);
    const auto& outter_indexes_expr = GetIndexesExpr(
        op_arg_path.op_steps.at(i), op2anchor2indexes_expr_signature);
    ADT_RETURN_IF_ERROR(outter_indexes_expr);
    const auto& composed_indexes_expr =
        builder.Compose(outter_indexes_expr.GetOkValue(), indexes_expr);
    ADT_RETURN_IF_ERROR(composed_indexes_expr);
    indexes_expr = composed_indexes_expr.GetOkValue();
  }
  return TrackedIndexesTransform{indexes_expr};
}

adt::Result<OpIndexesTransformSignature>
InferOpIndexesTransformSignatureByOpArgPath(
    const pir::Operation* yield_op,
    const OpArgTo<OpArgPath>& op_arg2path,
    const Op2Anchor2IndexesExprSignature& op2anchor2indexes_expr_signature,
    int src_idx) {
  pexpr::index_expr::InputSignature<TrackedIndexesTransform> in_sig;
  in_sig.descriptors->reserve(yield_op->num_operands());
  for (int i = 0; i < yield_op->num_operands(); ++i) {
    OpArg in_arg{yield_op->operand(i)};
    const auto& iter = op_arg2path.find(in_arg);
    if (iter == op_arg2path.end()) {
      return TypeError{std::string() + "no infer path from arg " +
                       std::to_string(src_idx) + " to  arg " +
                       std::to_string(i) + " found."};
    }
    const OpArgPath& op_arg_path = iter->second;
    const auto& transform_descriptor = MakeIndexesTransformByPath(
        op_arg_path, op2anchor2indexes_expr_signature);
    ADT_RETURN_IF_ERROR(transform_descriptor);
    in_sig.descriptors->emplace_back(transform_descriptor.GetOkValue());
  }
  return OpIndexesTransformSignature{
      in_sig, pexpr::index_expr::OutputSignature<TrackedIndexesTransform>{}};
}

OpArgTo<pexpr::RecordableIndexClosure> InferRecordableIndexClosuresByOpArgPaths(
    const pir::Operation* yield_op,
    const OpArgPaths& op_arg_paths,
    const Op2Anchor2IndexesExprSignature& op2anchor2indexes_expr_signature) {
  OpArgTo<pexpr::RecordableIndexClosure> ret;
  VisitOpInArg(yield_op, [&](const OpArg& in_arg, int, src_idx) {
    const auto& iter = op_arg_paths->paths.find(in_arg);
    if (iter == op_arg_paths->paths.end()) {
      return;
    }
    const auto& dst_op_arg_paths = iter->second;
    adt::Result<OpIndexesTransformSignature> indexes_transform_sig =
        InferOpIndexesTransformSignatureByOpArgPath(
            yield_op,
            dst_op_arg_paths,
            op2anchor2indexes_expr_signature,
            src_idx);
    if (indexes_transform_sig.HasError()) {
      return;
    }
    ret[in_arg] =
        pexpr::RecordableIndexClosure{indexes_transform_sig.GetOkValue()};
  });
  return std::move(ret);
}

adt::Result<OpArgTo<pexpr::RecordableIndexClosure>> GetOpArg2IndexClosure(
    const ::common::TopoWalker<OpOrArg>& walker,
    const cinn::dialect::FusionOp& fusion_op,
    const Op2Anchor2IndexesExprSignature& op2anchor2indexes_expr_signature) {
  const auto yield_op = GetYieldOp(fusion_op);
  if (!yield_op.has_value()) {
    return adt::errors::InvalidArgument {
      "block of fusion op has no yield op."
    }
  }
  adt::Result<OpOrArgPaths> op_or_arg_paths =
      GetPathsBetweenYieldOpInArgs(walker, yield_op.value());
  ADT_RETURN_IF_ERROR(op_or_arg_paths);
  adt::Result<OpArgPaths> op_arg_paths =
      ConvertPathsToOpArgPaths(op_or_arg_paths.GetOkValue());
  ADT_RETURN_IF_ERROR(op_arg_paths);
  return InferRecordableIndexClosuresByOpArgPaths(
      yield_op.value(),
      op_arg_paths.GetOkValue(),
      op2anchor2indexes_expr_signature);
}

}  // namespace

adt::Result<FusionDescriptor> GetFusionDescriptor(
    const cinn::dialect::FusionOp& op,
    ShapeConstraintIRAnalysis* shape_analysis) {
  ::common::TopoWalker<OpOrArg> base_walker = GetOpOrArgTopoWalker(op);
  if (FLAGS_enable_debug_ap) {
    CheckReachableToAllOps("Check base TopoWalker<OpOrArg>", base_walker, op);
  }
  auto interpreter = std::make_shared<IndexExprInterpreter>();
  auto DimExprs4Value = MakeDimExpr4Value(op, shape_analysis);
  auto Nice2IndexLambdas4Op = MakeNice2IndexLambdas4Op(op);
  auto IndexClosure4Op = MakeIndexClosure4Op(
      interpreter, DimExprs4Value, Nice2IndexLambdas4Op, op);
  OpArg2OpIndexesExprSignatureInferrer inferer(
      op, DimExprs4Value, IndexClosure4Op);
  auto op2anchor2indexes_expr_signature =
      inferer.GetOp2Anchor2IndexesExprSignature(op);
  auto anchorable_walker =
      GetAnchorableTopoWalker(base_walker, op2anchor2indexes_expr_signature);
  if (FLAGS_enable_debug_ap) {
    CheckReachableToAllOps(
        "Check anchorable TopoWalker<OpOrArg>", anchorable_walker, op);
  }
  adt::Result<OpArgTo<pexpr::RecordableIndexClosure>>
      yield_op_arg2index_closure = GetOpArg2IndexClosure(
          anchorable_walker, op, op2anchor2indexes_expr_signature);
  ADT_RETURN_IF_ERROR(yield_op_arg2index_closure);
  TrivialFusionDescriptor trivial_fusion_descriptor{
      op2anchor2indexes_expr_signature,
      yield_op_arg2index_closure.GetOkValue()};
  return FusionDescriptor{trivial_fusion_descriptor};
}

}  // namespace ap
