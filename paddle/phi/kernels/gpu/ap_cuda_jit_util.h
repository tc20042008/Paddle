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

#pragma once

#include "paddle/cinn/backends/nvrtc/nvrtc_util.h"
#include "paddle/cinn/runtime/cuda/cuda_module.h"

namespace ap {

const int kCUDAMaxCards{8};

/**
 * The CUDA module, helps to compile CUDA codes and fetch symbols.
 * Currently, it is a wrapper of NVRTC.
 */
class CUDAModule {
 public:
  enum class Kind {
    PTX = 0,
    CUBIN = 1,
  };

  CUDAModule(const std::string& data, Kind kind);

  void LaunchKernel(int device_id,
                    const std::string& func_name,
                    dim3 gridDim,
                    dim3 blockDim,
                    void** args,
                    size_t share_memory_size = 0,
                    CUstream stream = nullptr);

  //! Get a function.
  CUfunction GetFunction(int device_id, const std::string& func_name);

  //! Get a function by CudaGetDevice
  CUfunction GetFunction(const std::string& func_name);

  //! Get a global variable.
  CUdeviceptr GetGlobal(int device_id, const std::string& name, size_t nbytes);

  ~CUDAModule();

 private:
  //! The input data.
  std::string data_;
  //! Kind of the input.
  Kind kind_;
  //! To make parallel, we prepare one module for each card.
  std::vector<CUmodule> module_per_card_{kCUDAMaxCards, nullptr};
  std::string cuda_source_;
  std::mutex mutex_;

  CUdevice device_;
  CUcontext context_;
  int num_devices_{0};
};

/**
 * An helper class to call NVRTC. Input CUDA device source code, get PTX string.
 */
class Compiler {
 public:
  Compiler();

  /**
   * Compile the \p code and get PTX string.
   * @param code The CUDA source code.
   * @param include_headers Whether to include the headers of CUDA and CINN
   * runtime modules.
   * @return Compiled PTX code string.
   */
  std::string operator()(const std::string& code, bool include_headers = true);

  /** Compile into cubin or not
   * @return Compile into cubin or not.
   */
  bool compile_to_cubin();

 private:
  /**
   * Get the directories of CUDA's header files.
   * @return list of header file directories.
   */
  std::vector<std::string> FindCUDAIncludePaths();

  /**
   * Get the directories of CINN runtime's header files.
   * @return list of header file directories.
   */
  std::vector<std::string> FindCINNRuntimeIncludePaths();

  /**
   * Compile CUDA source code and get PTX or CUBIN.
   * @param code source code string.
   * @return PTX or CUBIN string.
   */
  std::string CompileCudaSource(const std::string& code, bool include_headers);

  /**
   * whether to compile the source code into cubin, only works with cuda version
   * > 11.1
   */
  bool compile_to_cubin_{false};

  // compile with nvcc
  std::string CompileWithNvcc(const std::string&);

  // compile to ptx
  void CompileToPtx();
  // compile to cubin
  void CompileToCubin();
  std::string GetDeviceArch();

  std::string ReadFile(const std::string&, std::ios_base::openmode);

  std::string prefix_name_{""};
};

}  // namespace ap
