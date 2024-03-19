#pragma once

#include "paddle/cinn/adt/stack.h"
#include "paddle/cinn/frontend/dag_gen_requirement.h"
#include "paddle/cinn/frontend/dag_gen_instruction.h"


namespace cinn::frontend {

class DAGGenerator {
 public:
  virtual ~DAGGenerator() = default;
  DAGGenerator(const DAGGenerator&) = default;
  DAGGenerator(DAGGenerator&&) = default;

  // Instructions generating sink nodes of DAG are on put the top of stack.
  virtual adt::Stack<const DAGGenInstruction> Generate(
    const adt::Stack<const DAGGenInstruction>& core_instruction) = 0;

 protected:
  explicit DAGGenerator(const DAGGenRequirement& requirement)
    : requirement_(requirement) {}
  const DAGGenRequirement requirement_;
};

std::unique_ptr<DAGGenerator> MakeDefaultDAGGenerator(
    const DAGGenRequirement& requirement);

}