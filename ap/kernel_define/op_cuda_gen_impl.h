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

#include "ap/adt/adt.h"
#include "ap/kernel_define/ir_op.h"
#include "ap/kernel_define/op_code_gen_ctx.h"

namespace ap::kernel_define {

template <typename IrNodeT>
struct OpCudaCodeGenImpl {
  adt::Result<std::string> CodeGen(const OpCodeGenCtx<IrNodeT>& op_code_gen_ctx,
                                   const IrOp<IrNodeT>& ir_op);
};

}  // namespace ap::kernel_define
