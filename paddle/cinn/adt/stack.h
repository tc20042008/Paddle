#pragma once

#include "glog/logging.h"
#include "paddle/cinn/adt/slist.h"

namespace cinn::adt {

template <typename T>
class Stack final {
 public:
  explicit Stack() : slist_(Nothing{}) {}
  Stack(const Stack& stack) = default;
  Stack(Stack&& stack) = default;

  bool empty() const {
    return slist_.template Has<Nothing>();
  }

  const T& top() const {
    CHECK(!empty());
    const auto& [data, _] = slist_.template Get<Cons<T, SList<T>>>().tuple();
    return data;
  }

  void pop() {
    CHECK(!empty());
    const auto& [_, next] = slist_.template Get<Cons<T, SList<T>>>().tuple();
    slist_ = next;
  }

  void push(const T& data) {
    slist_ = Cons<T, SList<T>>{data, slist_};
  }

 private:
  SList<T> slist_;
};

}