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

#include "paddle/pir/include/dialect/pexpr/adt.h"

namespace pexpr {

using adt::errors::AttributeError;
using adt::errors::Error;
using adt::errors::IndexError;
using adt::errors::InvalidArgumentError;
using adt::errors::NameError;
using adt::errors::NotImplementedError;
using adt::errors::RuntimeError;
using adt::errors::SyntaxError;
using adt::errors::TypeError;
using adt::errors::ValueError;
using adt::errors::ZeroDivisionError;

template <typename T>
using Result = adt::Result<T>;

}  // namespace pexpr
