#pragma once 

#include <variant>

namespace cinn::frontend {

#define FOR_EACH_DAG_GEN_TYPE(_macro)   \
  _macro(Nope)                          \
  _macro(AddSinkTensor)                 \
  _macro(AddUnaryUpstreamOp)            \
  _macro(AddBinaryUpstreamOp)           \
  _macro(InsertBinaryUpstreamOp)        \
  _macro(AddBinaryCloneUpstream)        \
  _macro(MarkFinalSourceTensor)

#define DECLARE_DAG_GEN_TYPE(dag_gen_type) \
template <typename T>                      \
struct dag_gen_type {                      \
};
FOR_EACH_DAG_GEN_TYPE(DECLARE_DAG_GEN_TYPE);
#undef DECLARE_DAG_GEN_TYPE

template <typename T>
using DAGGenType =
    std::variant<Nope<T>,
                 AddSinkTensor<T>,
                 AddUnaryUpstreamOp<T>,
                 AddBinaryUpstreamOp<T>,
                 InsertBinaryUpstreamOp<T>,    // append to core DAG
                 AddBinaryCloneUpstream<T>,    // modify core DAG.
                 MarkFinalSourceTensor<T>>;

}