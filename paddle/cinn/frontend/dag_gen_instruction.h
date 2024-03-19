#pragma once

#include "paddle/cinn/frontend/dag_gen_type.h"

namespace cinn::frontend {

struct DAGGenInstruction;

using NopeGenInstruction = Nope<DAGGenInstruction>;

using AddSinkTensorGenInstruction = AddSinkTensor<DAGGenInstruction>;

template <>
struct AddUnaryUpstreamOp<DAGGenInstruction> {
  int source_tensor_index;
  std::string dag_tag;
};
using AddUnaryUpstreamOpGenInstruction = AddUnaryUpstreamOp<DAGGenInstruction>;

template <>
struct AddBinaryUpstreamOp<DAGGenInstruction> {
  int source_tensor_index;
  std::string dag_tag;
};
using AddBinaryUpstreamOpGenInstruction = AddBinaryUpstreamOp<DAGGenInstruction>;

template <>
struct InsertBinaryUpstreamOp<DAGGenInstruction> {
  int source_tensor_index;
  std::string dag_tag;
};
using InsertBinaryUpstreamOpGenInstruction =
    InsertBinaryUpstreamOp<DAGGenInstruction>;

template <>
struct AddBinaryCloneUpstream<DAGGenInstruction> {
  int lhs_source_tensor_index;
  int rhs_source_tensor_index;
};
using AddBinaryCloneUpstreamGenInstruction =
    AddBinaryCloneUpstream<DAGGenInstruction>;

template <>
struct MarkFinalSourceTensor<DAGGenInstruction> {
  int source_tensor_index;
};
using MarkFinalSourceTensorGenInstruction =
    MarkFinalSourceTensor<DAGGenInstruction>;

// Generate sinks before sources.
// Generate DAG reversely by DAGGenInstruction.
struct DAGGenInstruction final : public DAGGenType<DAGGenInstruction> {
  using DAGGenType<DAGGenInstruction>::DAGGenType;
  
  const DAGGenType<DAGGenInstruction>& variant() const {
    return static_cast<const DAGGenType<DAGGenInstruction>&>(*this);
  }
};

}