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

namespace pexpr {

inline constexpr const char* kBuiltinNothing() { return "None"; }
inline constexpr const char* kBuiltinIf() { return "if"; }
inline constexpr const char* kBuiltinId() { return "__builtin_identity__"; }
inline constexpr const char* kBuiltinList() { return "__builtin_list__"; }
inline constexpr const char* kBuiltinGetAttr() { return "__builtin_getattr__"; }
inline constexpr const char* kBuiltinGetItem() { return "__builtin_getitem__"; }
inline constexpr const char* kBuiltinApply() { return "__builtin_apply__"; }
inline constexpr const char* kBuiltinReturn() { return "__builtin_return__"; }

}  // namespace pexpr
