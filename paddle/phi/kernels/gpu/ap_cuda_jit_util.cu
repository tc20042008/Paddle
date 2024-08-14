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

#include "paddle/phi/kernels/gpu/ap_cuda_jit_util.h"
#include <mutex>
#include <unordered_map>
#include "glog/logging.h"
#include "jitify.hpp"  // NOLINT
#include "paddle/cinn/backends/nvrtc/nvrtc_util.h"
#include "paddle/cinn/runtime/cuda/cuda_module.h"
#include "paddle/common/enforce.h"

namespace ap {

CUDAModule::CUDAModule(const std::string& data, Kind kind)
    : data_(data), kind_(kind) {
  PADDLE_ENFORCE_NE(
      data.empty(), true, phi::errors::PreconditionNotMet("data is is empty!"));

  cudaGetDeviceCount(&num_devices_);
  PADDLE_ENFORCE_GT(
      num_devices_, 0, phi::errors::ResourceExhausted("No available devices!"));

  // TODO(Superjomn) Determine whether to initialize all the devices.
  int current_device_id;
  cudaGetDevice(&current_device_id);
  cudaSetDevice(current_device_id);
  cuDeviceGet(&device_, current_device_id);
  cuCtxGetCurrent(&context_);
  cuDevicePrimaryCtxRetain(&context_, device_);
  VLOG(5) << "Construct CUDAModule " << this
          << " in device: " << current_device_id;
}

void CUDAModule::LaunchKernel(int device_id,
                              const std::string& func_name,
                              dim3 gridDim,
                              dim3 blockDim,
                              void** args,
                              size_t share_memory_size,
                              CUstream stream) {
  VLOG(3) << "cuLaunchKernel with func_name : " << func_name
          << ", gridDim.x:" << gridDim.x << ", gridDim.y:" << gridDim.y
          << ", gridDim.z:" << gridDim.z << ", blockDim.x:" << blockDim.x
          << ", blockDim.y:" << blockDim.y << ", blockDim.z:" << blockDim.z
          << ", share_memory_size:" << share_memory_size;
  auto function = GetFunction(device_id, func_name);
  PADDLE_ENFORCE_NOT_NULL(
      function,
      phi::errors::NotFound(
          "%s function not found on device %d.", func_name, device_id));
  CUDA_DRIVER_CALL(cuLaunchKernel(function,
                                  gridDim.x,
                                  gridDim.y,
                                  gridDim.z,
                                  blockDim.x,
                                  blockDim.y,
                                  blockDim.z,
                                  share_memory_size,
                                  stream,
                                  args,
                                  nullptr));
}

CUfunction CUDAModule::GetFunction(const std::string& func_name) {
  int device_id;
  cudaGetDevice(&device_id);
  return this->GetFunction(device_id, func_name);
}

CUfunction CUDAModule::GetFunction(int device_id,
                                   const std::string& func_name) {
  VLOG(5) << "GetFunction : " << func_name << " with device_id : " << device_id;
  if (!module_per_card_[device_id]) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Compilation with parameters
    const size_t jit_num_options = 5;
    std::vector<CUjit_option> jit_options(jit_num_options);
    std::vector<void*> jit_opt_vals(jit_num_options);

    // set up size of compilation log buffer
    jit_options[0] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    size_t log_buffer_size = 1024;
    jit_opt_vals[0] = reinterpret_cast<void*>(log_buffer_size);

    // set up pointer to the compilation log buffer
    jit_options[1] = CU_JIT_ERROR_LOG_BUFFER;
    std::vector<char> log_buffer(log_buffer_size, '\0');
    jit_opt_vals[1] = log_buffer.data();

    int value = 1;
    // Specifies whether to create debug information in output (-g)
    jit_options[2] = CU_JIT_GENERATE_DEBUG_INFO;
    jit_opt_vals[2] = reinterpret_cast<void*>(value);

    // Generate verbose log messages
    jit_options[3] = CU_JIT_LOG_VERBOSE;
    jit_opt_vals[3] = reinterpret_cast<void*>(value);

    // Generate line number information (-lineinfo)
    jit_options[4] = CU_JIT_GENERATE_LINE_INFO;
    jit_opt_vals[4] = reinterpret_cast<void*>(value);

    bool can_use_nvcc_compiler = false;
    if (can_use_nvcc_compiler) {
      CUDA_DRIVER_CALL(
          cuModuleLoad(&module_per_card_[device_id], data_.c_str()));
    } else {
      CUDA_DRIVER_CALL(cuModuleLoadDataEx(&module_per_card_[device_id],
                                          data_.c_str(),
                                          jit_num_options,
                                          jit_options.data(),
                                          jit_opt_vals.data()));
    }
  }

  CUfunction func;
  CUDA_DRIVER_CALL(cuModuleGetFunction(
      &func, module_per_card_[device_id], func_name.c_str()));
  return func;
}

CUdeviceptr CUDAModule::GetGlobal(int device_id,
                                  const std::string& name,
                                  size_t nbytes) {
  if (!module_per_card_[device_id]) {
    std::lock_guard<std::mutex> lock(mutex_);
    bool can_use_nvcc_compiler = false;
    if (can_use_nvcc_compiler) {
      CUDA_DRIVER_CALL(
          cuModuleLoad(&module_per_card_[device_id], data_.c_str()));
    } else {
      CUDA_DRIVER_CALL(
          cuModuleLoadData(&module_per_card_[device_id], data_.c_str()));
    }
  }

  size_t _nbytes;
  CUdeviceptr global;
  CUDA_DRIVER_CALL(cuModuleGetGlobal(
      &global, &_nbytes, module_per_card_[device_id], name.c_str()));
  return global;
}

CUDAModule::~CUDAModule() {
  for (int i = 0; i < module_per_card_.size(); i++) {
    auto* module = module_per_card_[i];
    if (module) {
      CUDA_CALL(cudaSetDevice(i));
      CUDA_DRIVER_CALL(cuModuleUnload(module));
    }
  }
}

class HeaderGeneratorBase {
 public:
  virtual const size_t size() const = 0;
  virtual const std::vector<const char*>& headers() const = 0;
  virtual const std::vector<const char*>& include_names() const = 0;
};

class JitSafeHeaderGenerator : public HeaderGeneratorBase {
 public:
  static HeaderGeneratorBase& GetInstance();
  const size_t size() const;
  const std::vector<const char*>& headers() const override { return headers_; }
  const std::vector<const char*>& include_names() const override {
    return include_names_;
  }

 private:
  JitSafeHeaderGenerator();
  std::vector<const char*> headers_;
  std::vector<const char*> include_names_;
};

HeaderGeneratorBase& JitSafeHeaderGenerator::GetInstance() {
  static JitSafeHeaderGenerator instance;
  return instance;
}

const size_t JitSafeHeaderGenerator::size() const {
  PADDLE_ENFORCE_EQ(
      include_names_.size(),
      headers_.size(),
      phi::errors::InvalidArgument("Internal error in size of header files."));
  return include_names_.size();
}

JitSafeHeaderGenerator::JitSafeHeaderGenerator() {
  const auto& headers_map = ::jitify::detail::get_jitsafe_headers_map();
  for (auto& pair : headers_map) {
    include_names_.emplace_back(pair.first.data());
    headers_.emplace_back(pair.second.data());
  }
}

static bool TryLocatePath(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

static std::vector<std::string> GetNvidiaAllIncludePath(
    const std::string& nvidia_package_dir) {
  std::vector<std::string> include_paths;
  const std::string delimiter = "/";
  // Expand this list if necessary.
  const std::vector<std::string> sub_modules = {"cublas",
                                                "cudnn",
                                                "cufft",
                                                "cusparse",
                                                "cusolver",
                                                "cuda_nvrtc",
                                                "curand",
                                                "cuda_runtime"};
  for (auto& sub_module : sub_modules) {
    std::string path =
        nvidia_package_dir + delimiter + sub_module + delimiter + "include";
    include_paths.push_back(path);
  }
  return include_paths;
}

std::string Compiler::operator()(const std::string& code,
                                 bool include_headers) {
  bool can_use_nvcc_compiler = false;
  if (can_use_nvcc_compiler) {
    return CompileWithNvcc(code);
  }
  return CompileCudaSource(code, include_headers);
}

Compiler::Compiler() {
  // Do nothing.
}

bool Compiler::compile_to_cubin() { return compile_to_cubin_; }

std::vector<std::string> Compiler::FindCUDAIncludePaths() {
  const std::string delimiter = "/";
  std::string cuda_include_path;
  const char* cuda_path_env = std::getenv("CUDA_PATH");
  if (cuda_path_env != nullptr) {
    cuda_include_path += cuda_path_env;
    cuda_include_path += delimiter + "include";
    VLOG(4) << "FindCUDAIncludePaths from CUDA_PATH: " << cuda_include_path;
    return {cuda_include_path};
  }

#if defined(__linux__)

  cuda_include_path = "/usr/local/cuda/include";
  if (TryLocatePath(cuda_include_path)) {
    VLOG(4) << "FindCUDAIncludePaths from " << cuda_include_path;
    return {cuda_include_path};
  }
#endif
  std::stringstream ss;
  ss << "Cannot find cuda include path."
     << "CUDA_PATH is not set or CUDA is not installed in the default "
        "installation path."
     << "In other than linux, it is necessary to set CUDA_PATH.";
  PADDLE_THROW(phi::errors::Fatal(ss.str()));
  return {cuda_include_path};
}

std::vector<std::string> Compiler::FindCINNRuntimeIncludePaths() { return {}; }

std::string Join(const std::vector<std::string>& strs, const std::string& sep) {
  std::string ret;
  int i = 0;
  for (const auto& str : strs) {
    if (i++ > 0) {
      ret += sep;
    }
    ret += str;
  }
  return ret;
}

std::string UniqName(const std::string& prefix) {
  const size_t seq_no = [&] {
    static std::mutex mutex;
    static std::unordered_map<std::string, size_t> prefix2seq_no;
    static std::unique_lock<std::mutex> lock(mutex);
    return prefix2seq_no[prefix]++;
  }();
  return prefix + "_" + std::to_string(seq_no);
}

std::string Compiler::CompileCudaSource(const std::string& code,
                                        bool include_headers) {
  const auto& header_gen = JitSafeHeaderGenerator::GetInstance();
  std::vector<std::string> compile_options;
  std::vector<const char*> param_cstrings{};
  nvrtcProgram prog;
  std::string cc = "30";
  int major, minor;
  cudaError_t e1 =
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  cudaError_t e2 =
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);

  if (e1 == cudaSuccess && e2 == cudaSuccess) {
    cc = std::to_string(major) + std::to_string(minor);
  } else {
    LOG(WARNING) << "cannot detect compute capability from your device, "
                 << "fall back to compute_30.";
  }
  if (compile_to_cubin_) {
    compile_options.push_back("-arch=sm_" + cc);
    std::string enable_fmad = "false";
    compile_options.push_back("--fmad=" + enable_fmad);
  } else {
    compile_options.push_back("-arch=compute_" + cc);
  }
  compile_options.push_back("-std=c++14");
  compile_options.push_back("-default-device");

  if (include_headers) {  // prepare include headers
    auto cuda_headers = FindCUDAIncludePaths();
    auto cinn_headers = FindCINNRuntimeIncludePaths();
    std::vector<std::string> include_paths;
    for (auto& header : cuda_headers) {
      VLOG(5) << "add include-path: " << header;
      include_paths.push_back("--include-path=" + header);
    }
    for (auto& header : cinn_headers) {
      include_paths.push_back("--include-path=" + header);
    }
    compile_options.insert(
        std::end(compile_options), include_paths.begin(), include_paths.end());
  }

  for (const auto& option : compile_options) {
    param_cstrings.push_back(option.c_str());
  }
  VLOG(3) << "compile options: " << Join(compile_options, " ");
  NVRTC_CALL(nvrtcCreateProgram(&prog,
                                code.c_str(),
                                nullptr,
                                header_gen.size(),
                                header_gen.headers().data(),
                                header_gen.include_names().data()));
  nvrtcResult compile_res =
      nvrtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data());

  const auto& GetLog = [&]() {
    size_t log_size = 0;
    NVRTC_CALL(nvrtcGetProgramLogSize(prog, &log_size));
    std::string log;
    log.resize(log_size);
    NVRTC_CALL(nvrtcGetProgramLog(prog, &log[0]));
    return log;
  };

  {  // get log
    PADDLE_ENFORCE_EQ(
        compile_res,
        NVRTC_SUCCESS,
        phi::errors::Fatal("NVRTC compilation failed. "
                           "\n================[code]================\n" +
                           code + "\n================[log]================\n" +
                           GetLog()));
  }

  size_t size;
  std::string data;
  if (compile_to_cubin_) {
    NVRTC_CALL(nvrtcGetCUBINSize(prog, &size));
    data.resize(size);
    NVRTC_CALL(nvrtcGetCUBIN(prog, &data[0]));
  } else {
    NVRTC_CALL(nvrtcGetPTXSize(prog, &size));
    data.resize(size);
    NVRTC_CALL(nvrtcGetPTX(prog, &data[0]));
  }

  NVRTC_CALL(nvrtcDestroyProgram(&prog));
  return data;
}

std::string Compiler::CompileWithNvcc(const std::string& cuda_c) {
  // read dir source
  std::string dir = "./source";
  if (access(dir.c_str(), 0) == -1) {
    CHECK(mkdir(dir.c_str(), 7) != -1) << "Fail to mkdir " << dir;
  }

  // get unique prefix name
  prefix_name_ = dir + "/" + UniqName("rtc_tmp");

  auto cuda_c_file = prefix_name_ + ".cu";
  std::ofstream ofs(cuda_c_file, std::ios::out);
  CHECK(ofs.is_open()) << "Fail to open file " << cuda_c_file;
  ofs << cuda_c;
  ofs.close();

  CompileToPtx();
  CompileToCubin();

  return prefix_name_ + ".cubin";
}

// std::string Compiler::GetPtx() { return ReadFile(prefix_name_ + ".ptx",
// std::ios::in); }

void Compiler::CompileToPtx() {
  std::vector<std::string> include_dir = {};
  std::string include_dir_str = "";
  for (auto dir : include_dir) {
    if (include_dir_str.empty()) {
      include_dir_str = dir;
    } else {
      include_dir_str += ":" + dir;
    }
  }
  const std::string FLAGS_cinn_nvcc_cmd_path = "/usr/local/cuda/bin";
  std::string options = std::string("export PATH=") + FLAGS_cinn_nvcc_cmd_path +
                        std::string(":$PATH && nvcc -std=c++14 --ptx -O3 -I ") +
                        include_dir_str;
  options += " -arch=" + GetDeviceArch();
  options += " -o " + prefix_name_ + ".ptx";
  options += " " + prefix_name_ + ".cu";

  VLOG(2) << "Nvcc Compile Options : " << options;
  CHECK(system(options.c_str()) == 0) << options;
}

void Compiler::CompileToCubin() {
  const std::string FLAGS_cinn_nvcc_cmd_path = "/usr/local/cuda/bin";
  std::string options = std::string("export PATH=") + FLAGS_cinn_nvcc_cmd_path +
                        std::string(":$PATH && nvcc --cubin -O3");
  options += " -arch=" + GetDeviceArch();
  options += " -o " + prefix_name_ + ".cubin";
  options += " " + prefix_name_ + ".ptx";

  VLOG(2) << "Nvcc Compile Options : " << options;
  CHECK(system(options.c_str()) == 0) << options;
}

std::string Compiler::GetDeviceArch() {
  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) ==
          cudaSuccess &&
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) ==
          cudaSuccess) {
    return "sm_" + std::to_string(major) + std::to_string(minor);
  } else {
    LOG(WARNING) << "cannot detect compute capability from your device, "
                 << "fall back to compute_30.";
    return "sm_30";
  }
}

std::string Compiler::ReadFile(const std::string& file_name,
                               std::ios_base::openmode mode) {
  // open cubin file
  std::ifstream ifs(file_name, mode);
  CHECK(ifs.is_open()) << "Fail to open file " << file_name;
  ifs.seekg(std::ios::end);
  auto len = ifs.tellg();
  ifs.seekg(0);

  // read cubin file
  std::string file_data(len, ' ');
  ifs.read(&file_data[0], len);
  ifs.close();
  return std::move(file_data);
}

}  // namespace ap
