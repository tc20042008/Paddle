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

#include "ap/code_gen/op_cuda_gen_impl.h"
#include "ap/code_gen/value.h"

namespace ap::code_gen {

template <typename BirNode>
adt::Result<std::string> OpCudaCodeGen(
    const OpCodeGenCtx<BirNode>& op_code_gen_ctx, const IrOp<BirNode>& ir_op) {
  OpCudaCodeGenImpl<BirNode> impl{};
  ADT_LET_CONST_REF(code, impl.CodeGen(op_code_gen_ctx, ir_op));
  return code;
}

}  // namespace ap::code_gen
