#pragma once

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/nothing.h"

namespace cinn::adt {

template <typename T0, typename T1>
struct Cons : public Tuple<T0, T1> {
  using Tuple<T0, T1>::Tuple;
};

template <typename T>
DEFINE_ADT_UNION(SList,
                 Nothing,
                 Cons<T, SList<T>>);

}