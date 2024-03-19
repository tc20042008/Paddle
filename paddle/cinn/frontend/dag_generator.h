#pragma once

#include "paddle/cinn/adt/stack.h"
#include "paddle/cinn/frontend/dag_gen_requirement.h"
#include "paddle/cinn/frontend/dag_gen_instruction.h"


namespace cinn::frontend {

// Instructions generating sink nodes of DAG are on put the top of stack.
adt::Stack<const DAGGenInstruction> GenerateDAGInstructions(
    const DAGGenRequirement& requirement,
    const adt::Stack<const DAGGenInstruction>& core_instruction);

}