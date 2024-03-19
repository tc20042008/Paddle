#pragma once

#include <stack>
#include "paddle/cinn/frontend/dag_gen_requirement.h"
#include "paddle/cinn/frontend/dag_gen_instruction.h"


namespace cinn::frontend {

std::stack<DAGGenInstruction> GenerateDAGInstructions(
    const DAGGenRequirement& requirement,
    const std::stack<DAGGenInstruction>& core_instruction);

}