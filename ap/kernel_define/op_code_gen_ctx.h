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
#include "ap/axpr/type.h"
#include "ap/index_expr/index_tuple_expr.h"

namespace ap::kernel_define {

template <typename IrNodeT>
struct DefineCtxImpl;

using LocalVarName = std::string;

template <typename IrNodeT>
struct LocalVarBinding {
  using NativeIrValue = typename IrNodeT::native_value_type;

  LocalVarName local_var_name;
  NativeIrValue ir_value;
};

template <typename IrNodeT>
struct OpCodeGenCtxImpl {
  std::weak_ptr<DefineCtxImpl<IrNodeT>> define_ctx;

  index_expr::IndexTupleExpr loop_index_tuple_expr;

  std::vector<std::string> loop_var_names;

  std::vector<LocalVarBinding<IrNodeT>> local_var_binding;

  std::optional<LocalVarName> anchor_local_var_name;

  adt::Result<LocalVarName> GetAnchorLocalVarName() const {
    const auto& opt_anchor = GetAnchorLocalVarNameWithoutCheck();
    ADT_CHECK(opt_anchor.has_value());
    ADT_RETURN_IF_ERR(CheckAnchorLocalVarName(opt_anchor.value()));
    return opt_anchor.value();
  }

  bool operator==(const OpCodeGenCtxImpl& other) const {
    return this == &other;
  }

 private:
  std::optional<LocalVarName> GetAnchorLocalVarNameWithoutCheck() const {
    if (this->anchor_local_var_name.has_value()) {
      return this->anchor_local_var_name.value();
    }
    if (this->loop_var_names.size() != 1) {
      return std::nullopt;
    }
    return this->loop_var_names.at(0);
  }

  adt::Result<adt::Ok> CheckAnchorLocalVarName(
      const LocalVarName& anchor) const {
    for (const auto& [local_var_name, _] : this->loop_var_names) {
      if (anchor == local_var_name) {
        return adt::Ok{};
      }
    }
    return adt::errors::ValueError{
        "anchor local var name not found in local var name bindings."};
  }
};

template <typename IrNodeT>
DEFINE_ADT_RC(OpCodeGenCtx, OpCodeGenCtxImpl<IrNodeT>);

}  // namespace ap::kernel_define
