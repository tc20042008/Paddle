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

#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/lower_fusion_op_to_ap_unary_prototype_pass.h"
#include "paddle/cinn/hlir/dialect/operator/ir/fusion_descriptor.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"

namespace cinn::dialect::ir {

namespace {

class FusionOpPattern : public pir::OpRewritePattern<cinn::dialect::FusionOp> {
 public:
  FusionOpPattern(
      ::pir::IrContext* context,
      const std::unordered_map<const pir::Operation*, ap::FusionDescriptor>*
          op2fusion_descriptor)
      : pir::OpRewritePattern<cinn::dialect::FusionOp>(context),
        op2fusion_descriptor_(op2fusion_descriptor) {}

  bool MatchAndRewrite(cinn::dialect::FusionOp fusion_op,
                       pir::PatternRewriter& rewriter) const override {
    const auto& opt_fusion_descriptor =
        [&]() -> std::optional<ap::FusionDescriptor> {
      const auto& iter = op2fusion_descriptor_->find(fusion_op);
      if (iter == op2fusion_descriptor_->end()) {
        return std::nullopt;
      }
      return iter->second;
    }();
    if (!opt_fusion_descriptor.has_value()) {
      return false;
    }
    TODO return false;
  }

 private:
  const std::unordered_map<const pir::Operation*, ap::FusionDescriptor>*
      op2fusion_descriptor_;  // not owned
};

std::unordered_map<const pir::Operation*, ap::FusionDescriptor>
GetOp2FusionDescriptor(pir::Operation* op) {
  std::unordered_map<const pir::Operation*, ap::FusionDescriptor> ret;
  op->Walk([&](pir::Operation* op) {
    if (!op->isa<cinn::dialect::FusionOp>()) {
      return;
    }
    const auto& opt_fusion_descriptor =
        GetFusionDescriptor(op->dyn_cast<cinn::dialect::FusionOp>());
    if (opt_fusion_descriptor.has_value()) {
      ret[op] = opt_fusion_descriptor.value();
    }
  });
  return ret;
}

class LowerFusionOpToApUnaryPrototypePass : public pir::PatternRewritePass {
 public:
  LowerFusionOpToApUnaryPrototypePass()
      : pir::PatternRewritePass("lower_fusion_op_to_ap_unary_prototype", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    context->GetOrRegisterDialect<cinn::dialect::RuntimeDialect>();
    context->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();

    pir::RewritePatternSet ps(context);
    ps.Add<FusionOpPattern>(context, &fusion_descriptor_);
    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    if (op->isa<pir::ModuleOp>()) {
      VLOG(5) << "start to pre-analysis all fusion ops in ModuleOp with static "
                 "shape mode.";
      fusion_descriptor_ = GetOp2FusionDescriptor(op);
    }
    return op->num_regions() > 0;
  }

 private:
  mutable std::unordered_map<const pir::Operation*, ap::FusionDescriptor>
      op2fusion_descriptor_;
};

}  // namespace

std::unique_ptr<::pir::Pass> CreateLowerFusionOpToApUnaryPrototypePass() {
  return std::make_unique<LowerFusionOpToApUnaryPrototypePass>();
}

}  // namespace cinn::dialect::ir
