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

#include "ap/kernel_dispatch/ap_unary_kernel.h"

#include <mutex>
#include <unordered_map>
#include "glog/logging.h"
#include "jitify.hpp"  // NOLINT
#include "paddle/common/enforce.h"

#include "ap/axpr/anf_expr_util.h"
#include "ap/kernel_dispatch/ap_cuda_jit_util.h"
#include "paddle/cinn/backends/nvrtc/nvrtc_util.h"
#include "paddle/cinn/runtime/cuda/cuda_module.h"
#include "paddle/phi/core/ap/kernel_define_helper.h"
#include "paddle/phi/core/ap/kernel_dispatch_helper.h"

namespace ap {

using MakeCoreExprT = adt::Result<ap::axpr::Lambda<ap::axpr::CoreExpr>> (*)(
    const std::string& json_str);

adt::Result<ap::axpr::Lambda<ap::axpr::CoreExpr>> ConvertToCoreExpr(
    const std::string& json_str) {
  const auto& anf_expr = ap::axpr::MakeAnfExprFromJsonString(json_str);
  ADT_RETURN_IF_ERR(anf_expr);
  const auto& core_expr =
      ap::axpr::ConvertAnfExprToCoreExpr(anf_expr.GetOkValue());
  if (!core_expr.Has<ap::axpr::Atomic<ap::axpr::CoreExpr>>()) {
    return adt::errors::TypeError{
        std::string() + "json_str can not be converted to atomic AnfExpr."};
  }
  const auto& atomic = core_expr.Get<ap::axpr::Atomic<ap::axpr::CoreExpr>>();
  if (!atomic.Has<ap::axpr::Lambda<ap::axpr::CoreExpr>>()) {
    return adt::errors::TypeError{
        std::string() + "json_str can not be converted to lambda AnfExpr."};
  }
  return atomic.Get<ap::axpr::Lambda<ap::axpr::CoreExpr>>();
}

template <MakeCoreExprT MakeCoreExpr>
adt::Result<ap::axpr::Lambda<ap::axpr::CoreExpr>> CacheCoreExpr(
    const std::string& json_str) {
  static std::unordered_map<std::string,
                            adt::Result<ap::axpr::Lambda<ap::axpr::CoreExpr>>>
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
      const std::shared_ptr<ap::paddle::CUDAModule>& cuda_module_val)
      : CudaModule(), module_(module_val), cuda_module_(cuda_module_val) {}

  ApUnaryCudaModuleImpl& operator=(const ApUnaryCudaModuleImpl& other) {
    this->module_ = other.module_;
    this->cuda_module_ = other.cuda_module_;
    return *this;
  }

  ApUnaryCudaModuleImpl& operator=(ApUnaryCudaModuleImpl&& other) {
    this->module_ = std::move(other.module_);
    this->cuda_module_ = std::move(other.cuda_module_);
    return *this;
  }

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
  std::shared_ptr<ap::paddle::CUDAModule> cuda_module_;
};
DEFINE_ADT_RC(ApUnaryCudaModule, ApUnaryCudaModuleImpl);

adt::Result<std::shared_ptr<ap::paddle::CUDAModule>> MakeBackendCudaModule(
    const Module& m) {
  ap::paddle::Compiler compiler;
  const std::string& source_code = m->source_code->source_code;
  auto ptx = compiler(source_code);
  ADT_CHECK(!ptx.empty()) << adt::errors::RuntimeError{
      std::string() + "Compilation failed. source_code: " + source_code};
  return std::make_shared<ap::paddle::CUDAModule>(
      ptx, ap::paddle::CUDAModule::Kind::PTX);
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
  ADT_LET_CONST_REF(kernel_definer_core_expr,
                    MakeOrGetCoreExpr(kernel_definer_lambda));
  phi::KernelDefineHelper helper{};
  ADT_LET_CONST_REF(
      m, helper.InterpretKernelDefineLambda(kernel_definer_core_expr));
  ADT_LET_CONST_REF(cuda_module, MakeBackendCudaModule(m));
  return ApUnaryCudaModule(m, cuda_module);
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
    ret->emplace_back(ConstTensor<Val>{tensor_data, dims});
  }
  return ret;
}

adt::List<Val> MakeMutableTensors(std::vector<phi::DenseTensor*>* ys) {
  adt::List<Val> ret;
  ret->reserve(ys->size());
  for (auto* y : *ys) {
    MutableTensorData tensor_data{y};
    adt::List<Val> dims{MakeTensorDims(*y)};
    ret->emplace_back(MutableTensor<Val>{tensor_data, dims});
  }
  return ret;
}

using FuncName2ArgTypes =
    std::unordered_map<std::string, adt::List<kernel_define::ArgType>>;
FuncName2ArgTypes MakeFuncName2ArgTypes(const kernel_define::Module& m) {
  auto GetArgTypes = [&](const auto& declare) { return declare->arg_types; };
  FuncName2ArgTypes ret;
  for (const auto& declare : *m->func_declares) {
    ret[declare->func_id] = GetArgTypes(declare);
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
  ADT_LET_CONST_REF(cuda_module,
                    kernel_define::MakeOrGetApUnaryCudaModule(
                        kernel_definer_lambda, define_ctx_maker_lambda));
  adt::List<Val> inputs = MakeConstTensors(xs);
  adt::List<Val> outputs = MakeMutableTensors(&outs);
  DispatchRawCtx<Val> raw_ctx{inputs,
                              outputs,
                              cuda_module.shared_ptr(),
                              MakeFuncName2ArgTypes(cuda_module->GetModule())};
  ADT_LET_CONST_REF(lambda, MakeOrGetCoreExpr(kernel_dispatcher_lambda));
  ADT_LET_CONST_REF(ctx_maker_lambda,
                    MakeOrGetCoreExpr(dispatch_ctx_maker_lambda));
  phi::KernelDispatchHelper helper{};
  ADT_RETURN_IF_ERR(helper.Interpret(lambda, ctx_maker_lambda, raw_ctx));
  return adt::Ok{};
}

}  // namespace kernel_dispatch

}  // namespace ap
