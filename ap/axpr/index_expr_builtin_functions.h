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
#include "ap/axpr/builtin_functions.h"
#include "ap/axpr/index_expr_value.h"

namespace ap::axpr::index_expr {

Result<Val> MakePtrGetItem(const Val&, const std::vector<Val>& args);

Result<Val> MakeIndexExprBroadcastMask(const Val&,
                                       const std::vector<Val>& args);

Result<Val> MakeSlice(const Val&, const std::vector<Val>& args);

Result<Val> MakeIndexExprSlice(const Val&, const std::vector<Val>& args);

Result<Val> MakeIndexExprAffine(const Val&, const std::vector<Val>& args);

Result<Val> MakeDisjointUnion(const Val&, const std::vector<Val>& args);

Result<Val> MakeIndexTupleExprPermute(const Val&, const std::vector<Val>& args);

Result<Val> MakeIndexTupleExprReshape(const Val&, const std::vector<Val>& args);

Result<Val> MakeIndexTupleExprTransform(const ApplyT<Val>& Apply,
                                        const Val& obj,
                                        const std::vector<Val>& args);

Result<Val> MakeOpIndexTupleExprSignature(const Val&,
                                          const std::vector<Val>& args);

Result<Val> MakeInIndexTupleExprSignature(const Val&,
                                          const std::vector<Val>& args);

Result<Val> MakeOutIndexTupleExprSignature(const Val&,
                                           const std::vector<Val>& args);

}  // namespace ap::axpr::index_expr
