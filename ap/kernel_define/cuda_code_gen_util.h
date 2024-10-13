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

#include "ap/kernel_define/op_cuda_gen_impl.h"
#include "ap/kernel_define/value.h"

namespace ap::kernel_define {

template <typename IrNodeT>
adt::Result<std::string> OpCudaCodeGen(
    const OpCodeGenCtx<IrNodeT>& op_code_gen_ctx, const IrOp<IrNodeT>& ir_op) {
  OpCudaCodeGenImpl<IrNodeT> impl{};
  ADT_LET_CONST_REF(code, impl.CodeGen(op_code_gen_ctx, ir_op));
  return code;
}

}  // namespace ap::kernel_define
