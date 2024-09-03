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

#include "paddle/phi/core/ap/ap_unary_kernel.h"

#include <mutex>
#include <unordered_map>
#include "glog/logging.h"
#include "jitify.hpp"  // NOLINT
#include "paddle/common/enforce.h"

#include "paddle/cinn/backends/nvrtc/nvrtc_util.h"
#include "paddle/cinn/runtime/cuda/cuda_module.h"
#include "paddle/phi/common/ap/define_ctx_value.h"
#include "paddle/phi/common/ap/dispatch_ctx_value.h"
#include "paddle/phi/common/ap/kernel_definer_interpreter.h"
#include "paddle/phi/common/ap/kernel_dispatcher_interpreter.h"
#include "paddle/phi/core/ap/ap_cuda_jit_util.h"
#include "paddle/pir/include/dialect/pexpr/anf_expr_util.h"

namespace ap {

using MakeCoreExprT = adt::Result<pexpr::Lambda<pexpr::CoreExpr>> (*)(
    const std::string& json_str);

adt::Result<pexpr::Lambda<pexpr::CoreExpr>> ConvertToCoreExpr(
    const std::string& json_str) {
  const auto& anf_expr = pexpr::MakeAnfExprFromJsonString(json_str);
  ADT_RETURN_IF_ERROR(anf_expr);
  const auto& core_expr =
      pexpr::ConvertAnfExprToCoreExpr(anf_expr.GetOkValue());
  if (!core_expr.Has<pexpr::Atomic<pexpr::CoreExpr>>()) {
    return adt::errors::TypeError{
        std::string() + "json_str can not be converted to atomic AnfExpr."};
  }
  const auto& atomic = core_expr.Get<pexpr::Atomic<pexpr::CoreExpr>>();
  if (!atomic.Has<pexpr::Lambda<pexpr::CoreExpr>>()) {
    return adt::errors::TypeError{
        std::string() + "json_str can not be converted to lambda AnfExpr."};
  }
  return atomic.Get<pexpr::Lambda<pexpr::CoreExpr>>();
}

template <MakeCoreExprT MakeCoreExpr>
adt::Result<pexpr::Lambda<pexpr::CoreExpr>> CacheCoreExpr(
    const std::string& json_str) {
  static std::unordered_map<std::string,
                            adt::Result<pexpr::Lambda<pexpr::CoreExpr>>>
      json_str2cache;
  static std::mutex mutex;
  std::unique_lock<std::mutex> lock(mutex);
  auto iter = json_str2cache.find(json_str);
  if (iter == json_str2cache.end()) {
    const auto& core_expr = MakeCoreExpr(json_str);
    iter = json_str2cache.emplace(json_str, core_expr).first;
  }
  return iter->second;
}

constexpr MakeCoreExprT MakeOrGetCoreExpr = &CacheCoreExpr<&ConvertToCoreExpr>;

namespace kernel_define {

class ApUnaryCudaModuleImpl : public kernel_dispatch::CudaModule {
 public:
  explicit ApUnaryCudaModuleImpl(
      const Module& module_val,
      const std::shared_ptr<ap::CUDAModule>& cuda_module_val)
      : CudaModule(), module_(module_val), cuda_module_(cuda_module_val) {}

  adt::Result<adt::Ok> LaunchCudaKernel(
      const std::string& func_name,
      int64_t num_blocks,
      int64_t num_threads,
      const std::vector<void*>& args) override {
    dim3 blocks_per_grid(num_blocks);
    dim3 threads_per_block(num_threads);
    std::vector<void*>* vec_ptr = const_cast<std::vector<void*>*>(&args);
    cuda_module_->LaunchKernel(
        0, func_name, blocks_per_grid, threads_per_block, vec_ptr->data());
    return adt::Ok{};
  }

  const Module& GetModule() const { return module_; }

 private:
  Module module_;
  std::shared_ptr<ap::CUDAModule> cuda_module_;
};
DEFINE_ADT_RC(ApUnaryCudaModule, ApUnaryCudaModuleImpl);

adt::Result<std::shared_ptr<ap::CUDAModule>> MakeBackendCudaModule(
    const Module& m) {
  ap::Compiler compiler;
  const std::string& source_code = m->source_code->source_code;
  auto ptx = compiler(source_code);
  if (ptx.empty()) {
    return adt::errors::RuntimeError{
        std::string() + "Compilation failed. source_code: " + source_code};
  }
  return std::make_shared<ap::CUDAModule>(ptx, ap::CUDAModule::Kind::PTX);
}

using MakeCudaModuleT = adt::Result<ApUnaryCudaModule> (*)(
    const std::string& kernel_definer_lambda,
    const std::string& define_ctx_maker_lambda);

template <MakeCudaModuleT MakeCudaModule>
adt::Result<ApUnaryCudaModule> CacheCudaModule(
    const std::string& kernel_definer_lambda,
    const std::string& define_ctx_maker_lambda) {
  using CtxMaker2CudaModule =
      std::unordered_map<std::string, adt::Result<ApUnaryCudaModule>>;
  using Definer2CtxMaker2CudaModule =
      std::unordered_map<std::string, CtxMaker2CudaModule>;
  static Definer2CtxMaker2CudaModule definer2ctx_maker2cuda_module;
  static std::mutex mutex;
  std::unique_lock<std::mutex> lock(mutex);
  auto* ctx_maker2cuda_module =
      &definer2ctx_maker2cuda_module[kernel_definer_lambda];
  auto iter = ctx_maker2cuda_module->find(define_ctx_maker_lambda);
  if (iter == ctx_maker2cuda_module->end()) {
    const auto& cuda_module =
        MakeCudaModule(kernel_definer_lambda, define_ctx_maker_lambda);
    iter = ctx_maker2cuda_module->emplace(define_ctx_maker_lambda, cuda_module)
               .first;
  }
  return iter->second;
}

adt::Result<ApUnaryCudaModule> MakeApUnaryCudaModule(
    const std::string& kernel_definer_lambda,
    const std::string& define_ctx_maker_lambda) {
  KernelDefinerInterpreter interpreter;
  const auto& define_ctx_maker_core_expr =
      MakeOrGetCoreExpr(define_ctx_maker_lambda);
  ADT_RETURN_IF_ERROR(define_ctx_maker_core_expr);
  const pexpr::Lambda<pexpr::CoreExpr>& ctx_maker =
      define_ctx_maker_core_expr.GetOkValue();
  DefinerRawCtx raw_ctx{};
  const Result<Val>& ctx = interpreter.CallLambda(ctx_maker, raw_ctx);
  const auto& kernel_definer_core_expr =
      MakeOrGetCoreExpr(kernel_definer_lambda);
  ADT_RETURN_IF_ERROR(kernel_definer_core_expr);
  const pexpr::Lambda<pexpr::CoreExpr>& definer =
      kernel_definer_core_expr.GetOkValue();
  const Result<Val>& interpret_ret =
      interpreter.CallLambda(definer, ctx.GetOkValue());
  ADT_RETURN_IF_ERROR(interpret_ret);
  const Result<Module>& m =
      CastToCustomValue<Module>(interpret_ret.GetOkValue());
  ADT_RETURN_IF_ERROR(m);
  const auto& cuda_module = MakeBackendCudaModule(m.GetOkValue());
  ADT_RETURN_IF_ERROR(cuda_module);
  return ApUnaryCudaModule(m.GetOkValue(), cuda_module.GetOkValue());
}

constexpr MakeCudaModuleT MakeOrGetApUnaryCudaModule =
    &CacheCudaModule<&MakeApUnaryCudaModule>;

}  // namespace kernel_define

namespace kernel_dispatch {

adt::List<Val> MakeTensorDims(const phi::DenseTensor& tensor) {
  adt::List<Val> ret;
  ret->reserve(tensor.dims().size());
  for (int i = 0; i < tensor.dims().size(); ++i) {
    ret->emplace_back(Val{tensor.dims().at(i)});
  }
  return ret;
}

adt::List<Val> MakeConstTensors(
    const std::vector<const phi::DenseTensor*>& xs) {
  adt::List<Val> ret;
  ret->reserve(xs.size());
  for (const auto* x : xs) {
    ConstTensorData tensor_data{x};
    adt::List<Val> dims{MakeTensorDims(*x)};
    ret->emplace_back(CustomValue{ConstTensor<Val>{tensor_data, dims}});
  }
  return ret;
}

adt::List<Val> MakeMutableTensors(std::vector<phi::DenseTensor*>* ys) {
  adt::List<Val> ret;
  ret->reserve(ys->size());
  for (auto* y : *ys) {
    MutableTensorData tensor_data{y};
    adt::List<Val> dims{MakeTensorDims(*y)};
    ret->emplace_back(CustomValue{MutableTensor<Val>{tensor_data, dims}});
  }
  return ret;
}

using FuncName2ArgTypes =
    std::unordered_map<std::string, adt::List<kernel_define::ArgType>>;
FuncName2ArgTypes MakeFuncName2ArgTypes(const kernel_define::Module& m) {
  FuncName2ArgTypes ret;
  for (const auto& declare : *m->func_declares) {
    ret[declare->func_id] = declare->arg_types;
  }
  return ret;
}

adt::Result<adt::Ok> ApUnaryKernel(
    const std::vector<const phi::DenseTensor*>& xs,
    int num_outputs,
    const std::string& kernel_definer_lambda,
    const std::string& define_ctx_maker_lambda,
    const std::string& kernel_dispatcher_lambda,
    const std::string& dispatch_ctx_maker_lambda,
    std::vector<phi::DenseTensor*> outs) {
  KernelDispatcherInterpreter interpreter{};
  const adt::Result<kernel_define::ApUnaryCudaModule>& m =
      kernel_define::MakeOrGetApUnaryCudaModule(kernel_definer_lambda,
                                                define_ctx_maker_lambda);
  ADT_RETURN_IF_ERROR(m);
  const auto& cuda_module = m.GetOkValue();
  adt::List<Val> inputs = MakeConstTensors(xs);
  adt::List<Val> outputs = MakeMutableTensors(&outs);
  DispatchRawContext<Val> raw_ctx{
      inputs,
      outputs,
      cuda_module.shared_ptr(),
      MakeFuncName2ArgTypes(cuda_module->GetModule())};
  const auto& dispatch_ctx_maker_core_expr =
      MakeOrGetCoreExpr(dispatch_ctx_maker_lambda);
  ADT_RETURN_IF_ERROR(dispatch_ctx_maker_core_expr);
  const auto& ctx_maker_lambda = dispatch_ctx_maker_core_expr.GetOkValue();
  const auto& ctx = interpreter.CallLambda(ctx_maker_lambda, raw_ctx);
  ADT_RETURN_IF_ERROR(ctx);
  const adt::Result<pexpr::Lambda<pexpr::CoreExpr>>&
      kernel_dispatcher_core_expr = MakeOrGetCoreExpr(kernel_dispatcher_lambda);
  ADT_RETURN_IF_ERROR(kernel_dispatcher_core_expr);
  const pexpr::Lambda<pexpr::CoreExpr>& lambda =
      kernel_dispatcher_core_expr.GetOkValue();
  const adt::Result<Val>& dispatch_ret =
      interpreter.CallLambda(lambda, ctx.GetOkValue());
  ADT_RETURN_IF_ERROR(dispatch_ret);
  return adt::Ok{};
}

}  // namespace kernel_dispatch

}  // namespace ap
