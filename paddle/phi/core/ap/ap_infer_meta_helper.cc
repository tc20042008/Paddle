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

#include "paddle/phi/core/ap/ap_infer_meta_helper.h"
#include <mutex>
#include "ap/adt/adt.h"
#include "ap/axpr/anf_expr_util.h"
#include "ap/axpr/const_std_vector_ptr.h"
#include "ap/axpr/const_std_vector_ptr_method_class.h"
#include "ap/axpr/cps_expr_interpreter.h"
#include "ap/axpr/data_type.h"
#include "ap/axpr/data_type_method_class.h"
#include "ap/axpr/std_vector_ptr.h"
#include "ap/axpr/std_vector_ptr_method_class.h"
#include "ap/axpr/value.h"
#include "ap/axpr/value_method_class.h"
#include "ap/paddle/const_meta_tensor_ptr.h"
#include "ap/paddle/const_meta_tensor_ptr_method_class.h"
#include "ap/paddle/ddim.h"
#include "ap/paddle/ddim_method_class.h"
#include "ap/paddle/meta_tensor_ptr.h"
#include "ap/paddle/meta_tensor_ptr_method_class.h"

namespace ap::paddle {

template <typename ValueT>
using ValueImpl = axpr::ValueBase<ValueT,
                                  axpr::DataType,
                                  paddle::DDim,
                                  ConstMetaTensorPtr,
                                  MetaTensorPtr,
                                  const std::vector<ConstMetaTensorPtr>*,
                                  std::vector<MetaTensorPtr>*>;

struct Value : public ValueImpl<Value> {
  using ValueImpl<Value>::ValueImpl;
  DEFINE_ADT_VARIANT_METHODS(ValueImpl<Value>);

  static axpr::Object<Value> GetExportedTypes() {
    return axpr::GetObjectTypeName2Type<Value, axpr::DataType>();
  }
};

}  // namespace ap::paddle

namespace phi {

namespace {

using CoreExpr = ap::axpr::CoreExpr;
using Lambda = ap::axpr::Lambda<CoreExpr>;

adt::Result<adt::Ok> InferMetaByLambda(
    const Lambda& lambda,
    const std::vector<const MetaTensor*>* inputs,
    std::vector<MetaTensor*>* outputs) {
  ap::axpr::CpsExprInterpreter<ap::paddle::Value> interpreter{};
  ADT_RETURN_IF_ERR(interpreter.Interpret(lambda, {inputs, outputs}));
  return adt::Ok{};
}

adt::Result<Lambda> MakeLambda(const std::string& lambda_str) {
  ADT_LET_CONST_REF(anf_expr, ap::axpr::MakeAnfExprFromJsonString(lambda_str));
  const auto& core_expr = ap::axpr::ConvertAnfExprToCoreExpr(anf_expr);
  ADT_LET_CONST_REF(atomic,
                    core_expr.TryGet<ap::axpr::Atomic<ap::axpr::CoreExpr>>())
      << adt::errors::TypeError{
             std::string() +
             "lambda_str can not be converted to atomic AnfExpr."};
  ADT_LET_CONST_REF(lambda,
                    atomic.TryGet<ap::axpr::Lambda<ap::axpr::CoreExpr>>());
  return lambda;
}

using MakeLambdaT = adt::Result<Lambda> (*)(const std::string& lambda_str);

template <MakeLambdaT Make>
adt::Result<Lambda> CacheConvertResult(const std::string& lambda_str) {
  static std::unordered_map<std::string, adt::Result<Lambda>> cache;
  static std::mutex mutex;
  std::unique_lock<std::mutex> lock(mutex);
  auto iter = cache.find(lambda_str);
  if (iter == cache.end()) {
    iter = cache.emplace(lambda_str, Make(lambda_str)).first;
  }
  ADT_LET_CONST_REF(lambda, iter->second);
  return lambda;
}

constexpr MakeLambdaT CastToLambda = &CacheConvertResult<&MakeLambda>;

}  // namespace

adt::Result<adt::Ok> ApInferMetaHelper::InferMeta(
    const std::string& lambda_str,
    const std::vector<const MetaTensor*>* inputs,
    std::vector<MetaTensor*>* outputs) {
  ADT_LET_CONST_REF(lambda, CastToLambda(lambda_str));
  return InferMetaByLambda(lambda, inputs, outputs);
}

}  // namespace phi
