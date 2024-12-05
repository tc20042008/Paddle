// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include <functional>
#include <list>
#include <memory>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "glog/logging.h"
#include "paddle/common/overloaded.h"

namespace cinn {
namespace adt {

template <typename T>
struct Rc {
 public:
  Rc() : data_(std::make_shared<T>()) {}
  explicit Rc(const std::shared_ptr<T>& data) : data_(data) {}
  Rc(const Rc&) = default;
  Rc(Rc&&) = default;
  Rc& operator=(const Rc&) = default;
  Rc& operator=(Rc&&) = default;

  template <typename Arg,
            std::enable_if_t<
                !std::is_same_v<std::decay_t<Arg>, Rc> &&
                    !std::is_same_v<std::decay_t<Arg>, std::shared_ptr<T>>,
                bool> = true>
  explicit Rc(Arg&& arg) : data_(new T{std::forward<Arg>(arg)}) {}

  template <typename Arg0, typename Arg1, typename... Args>
  explicit Rc(Arg0&& arg0, Arg1&& arg1, Args&&... args)
      : data_(new T{std::forward<Arg0>(arg0),
                    std::forward<Arg1>(arg1),
                    std::forward<Args>(args)...}) {}

  T* operator->() { return data_.get(); }
  const T* operator->() const { return data_.get(); }

  T& operator*() { return *data_; }
  const T& operator*() const { return *data_; }

  bool operator==(const Rc& other) const {
    if (other.data_.get() == this->data_.get()) {
      return true;
    }
    return *other.data_ == *this->data_;
  }

  const std::shared_ptr<T>& shared_ptr() const { return data_; }

  const void* __adt_rc_shared_ptr_raw_ptr() const { return data_.get(); }

 private:
  std::shared_ptr<T> data_;
};

#define DEFINE_ADT_RC(class_name, ...)                      \
  struct class_name : public ::cinn::adt::Rc<__VA_ARGS__> { \
    using ::cinn::adt::Rc<__VA_ARGS__>::Rc;                 \
  };

#define DEFINE_ADT_VARIANT_METHODS(...)                                \
  DEFINE_ADT_VARIANT_METHODS_WHITOUT_TRYGET(__VA_ARGS__)               \
  template <typename __AlternativeT, int start_idx = 0>                \
  static constexpr bool IsMyAlternative() {                            \
    if constexpr (start_idx >= std::variant_size_v<__VA_ARGS__>) {     \
      return false;                                                    \
    } else {                                                           \
      using AlternativeT =                                             \
          typename std::variant_alternative_t<start_idx, __VA_ARGS__>; \
      if constexpr (std::is_same_v<AlternativeT, __AlternativeT>) {    \
        return true;                                                   \
      } else {                                                         \
        return IsMyAlternative<__AlternativeT, start_idx + 1>();       \
      }                                                                \
    }                                                                  \
  }

#define DEFINE_ADT_VARIANT_METHODS_WHITOUT_TRYGET(...)                 \
  DEFINE_MATCH_METHOD();                                               \
  const __VA_ARGS__& variant() const {                                 \
    return reinterpret_cast<const __VA_ARGS__&>(*this);                \
  }                                                                    \
  template <typename __ADT_T>                                          \
  bool Has() const {                                                   \
    return std::holds_alternative<__ADT_T>(variant());                 \
  }                                                                    \
  template <typename __ADT_T>                                          \
  const __ADT_T& Get() const {                                         \
    return std::get<__ADT_T>(variant());                               \
  }                                                                    \
  bool operator!=(const __VA_ARGS__& other) const {                    \
    return !(*this == other);                                          \
  }                                                                    \
  bool operator==(const __VA_ARGS__& other) const {                    \
    return std::visit(                                                 \
        [](const auto& lhs, const auto& rhs) {                         \
          if constexpr (std::is_same_v<std::decay_t<decltype(lhs)>,    \
                                       std::decay_t<decltype(rhs)>>) { \
            return lhs == rhs;                                         \
          } else {                                                     \
            return false;                                              \
          }                                                            \
        },                                                             \
        this->variant(),                                               \
        other);                                                        \
  }

template <typename T>
class List final {
 public:
  List(const List&) = default;
  List(List&&) = default;
  List& operator=(const List&) = default;
  List& operator=(List&&) = default;

  using value_type = T;

  explicit List() : vector_(std::make_shared<std::vector<T>>()) {}

  template <
      typename Arg,
      std::enable_if_t<!std::is_same_v<std::decay_t<Arg>, List>, bool> = true>
  explicit List(Arg&& arg)
      : vector_(std::make_shared<std::vector<T>>(
            std::vector<T>{std::forward<Arg>(arg)})) {}

  template <typename Arg0, typename Arg1, typename... Args>
  List(Arg0&& arg0, Arg1&& arg1, Args&&... args)
      : vector_(std::make_shared<std::vector<T>>(
            std::vector<T>{std::forward<Arg0>(arg0),
                           std::forward<Arg1>(arg1),
                           std::forward<Args>(args)...})) {}

  bool operator==(const List& other) const {
    if (&vector() == &other.vector()) {
      return true;
    }
    return vector() == other.vector();
  }

  bool operator!=(const List& other) const { return !(*this == other); }

  std::vector<T>& operator*() const { return *vector_; }
  std::vector<T>* operator->() const { return vector_.get(); }

  const std::vector<T>& vector() const { return *vector_; }

  const auto& Get(std::size_t idx) const { return vector_->at(idx); }

 private:
  std::shared_ptr<std::vector<T>> vector_;
};

#define DEFINE_ADT_TAG(TagName)                                             \
  template <typename T>                                                     \
  class TagName {                                                           \
   public:                                                                  \
    TagName() = default;                                                    \
    TagName(const TagName&) = default;                                      \
    TagName(TagName&&) = default;                                           \
    TagName& operator=(const TagName&) = default;                           \
    TagName& operator=(TagName&&) = default;                                \
                                                                            \
    bool operator==(const TagName& other) const {                           \
      return value_ == other.value();                                       \
    }                                                                       \
                                                                            \
    bool operator!=(const TagName& other) const {                           \
      return value_ != other.value();                                       \
    }                                                                       \
                                                                            \
    template <typename Arg,                                                 \
              std::enable_if_t<!std::is_same_v<std::decay_t<Arg>, TagName>, \
                               bool> = true>                                \
    explicit TagName(Arg&& value) : value_(value) {}                        \
                                                                            \
    const T& value() const { return value_; }                               \
                                                                            \
   private:                                                                 \
    T value_;                                                               \
  };

// Undefined = {}
struct Undefined final : public std::monostate {
  using std::monostate::monostate;
};

// Ok = {}
struct Ok final : public std::monostate {
  using std::monostate::monostate;
};

inline std::size_t hash_combine(std::size_t lhs, std::size_t rhs) {
  return lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
}

struct Nothing : public std::monostate {
  using std::monostate::monostate;
};

struct IdentityFunc : public std::monostate {
  using std::monostate::monostate;
};

template <typename T0, typename T1>
using EitherImpl = std::variant<T0, T1>;

template <typename T0, typename T1>
struct Either : public EitherImpl<T0, T1> {
  using EitherImpl<T0, T1>::EitherImpl;
  DEFINE_ADT_VARIANT_METHODS_WHITOUT_TRYGET(EitherImpl<T0, T1>);
};

template <typename T>
struct Maybe : public Either<T, Nothing> {
  using Either<T, Nothing>::Either;
};

}  // namespace adt
}  // namespace cinn
