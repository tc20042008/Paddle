#pragma once 

#include <string>
#include "paddle/common/enforce.h"
#include "paddle/cinn/frontend/dag_gen_type.h"

namespace cinn::frontend {

struct PickWeight {
  explicit PickWeight(float weight_val) : weight(weight_val) {
    if (this->weight < 0) {
      this->weight = 0;
    }
  }
  PickWeight(const PickWeight&) = default;
  PickWeight(PickWeight&&) = default;

  float weight;
};

struct DAGGenTypePickProbability {
  PickWeight nope;
  PickWeight add_sink_tensor;
  PickWeight add_unary_upstream_op;
  PickWeight add_binary_upstream_op;    // append to core DAG.
  PickWeight insert_binary_upstream_op; // modify core DAG.
  PickWeight add_binary_clone_upstream;
  PickWeight mark_final_source_tensor;
};

inline float GetWeight(
    const Nope<std::monostate>&,
    const DAGGenTypePickProbability& prob) {
  return prob.nope.weight;
}

inline float GetWeight(
    const AddSinkTensor<std::monostate>&,
    const DAGGenTypePickProbability& prob) {
  return prob.add_sink_tensor.weight;
}

inline float GetWeight(
    const AddUnaryUpstreamOp<std::monostate>&,
    const DAGGenTypePickProbability& prob) {
  return prob.add_unary_upstream_op.weight;
}

inline float GetWeight(
    const AddBinaryUpstreamOp<std::monostate>&,
    const DAGGenTypePickProbability& prob) {
  return prob.add_binary_upstream_op.weight;
}

inline float GetWeight(
    const InsertBinaryUpstreamOp<std::monostate>&,
    const DAGGenTypePickProbability& prob) {
  return prob.insert_binary_upstream_op.weight;
}

inline float GetWeight(
    const AddBinaryCloneUpstream<std::monostate>&,
    const DAGGenTypePickProbability& prob) {
  return prob.add_binary_clone_upstream.weight;
}

inline float GetWeight(
    const MarkFinalSourceTensor<std::monostate>&,
    const DAGGenTypePickProbability& prob) {
  return prob.mark_final_source_tensor.weight;
}

struct DAGGenRequirement {
  size_t max_width;
  size_t max_instructions;
  std::string dag_tag;
  DAGGenTypePickProbability pick_probability;
};

}