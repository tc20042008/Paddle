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

#include <string>
#include <vector>

#include "ap/adt/adt.h"

namespace phi {

class DenseTensor;

}

namespace ap::kernel_dispatch {

adt::Result<adt::Ok> ApUnaryKernel(
    const std::vector<const phi::DenseTensor*>& xs,
    int num_outputs,
    const std::string& kernel_define_lambda,
    const std::string& define_ctx_maker_lambda,
    const std::string& kernel_dispatcher_lambda,
    const std::string& dispatch_ctx_maker_lambda,
    std::vector<phi::DenseTensor*> outs);

}  // namespace ap::kernel_dispatch
