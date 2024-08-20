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

struct RuntimeError {
  std::string msg;

  bool operator==(const RuntimeError& other) const {
    return other.msg == this->msg;
  }
};

struct InvalidArgumentError {
  std::string msg;

  bool operator==(const InvalidArgumentError& other) const {
    return other.msg == this->msg;
  }
};

struct AttributeError {
  std::string msg;

  bool operator==(const AttributeError& other) const {
    return other.msg == this->msg;
  }
};

struct NameError {
  std::string msg;

  bool operator==(const NameError& other) const {
    return other.msg == this->msg;
  }
};

struct TypeError {
  std::string msg;

  bool operator==(const TypeError& other) const {
    return other.msg == this->msg;
  }
};

struct SyntaxError {
  std::string msg;

  bool operator==(const SyntaxError& other) const {
    return other.msg == this->msg;
  }
};

using ErrorBase = std::variant<RuntimeError,
                               InvalidArgumentError,
                               AttributeError,
                               NameError,
                               TypeError,
                               SyntaxError>;

struct [[nodiscard]] Error : public ErrorBase {
  using ErrorBase::ErrorBase;
  DEFINE_ADT_VARIANT_METHODS(ErrorBase);
};

template <typename T>
struct [[nodiscard]] Result : public Either<T, Error> {
  using Either<T, Error>::Either;
};

}  // namespace pexpr
