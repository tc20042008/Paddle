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

#include <ctime>
#include <thread>  // NOLINT
#include "gtest/gtest.h"

#include "ap/axpr/anf_expr_util.h"
#include "ap/axpr/cps_expr_interpreter.h"
#include "ap/axpr/lambda_expr_builder.h"
#include "ap/axpr/value_method_class.h"
#include "ap/drr/drr_value.h"
#include "ap/drr/drr_value_method_class.h"
#include "paddle/common/errors.h"

namespace ap::drr::tests {

using axpr::AnfExpr;
using axpr::Atomic;
using axpr::ConvertAnfExprToCoreExpr;
using axpr::CoreExpr;
using axpr::CpsExprInterpreter;
using axpr::Lambda;
using axpr::LambdaExprBuilder;

TEST(DrrValue, source_pattern) {
  LambdaExprBuilder lmbd{};
  AnfExpr lmbd_expr = lmbd.Lambda({"ctx"}, [&](auto& ctx) {
    return ctx.Var("ctx").Attr("source_pattern");
  });
  const auto& core_expr = ConvertAnfExprToCoreExpr(lmbd_expr);
  ASSERT_TRUE(core_expr.Has<Atomic<CoreExpr>>());
  const auto& atomic = core_expr.Get<Atomic<CoreExpr>>();
  ASSERT_TRUE(atomic.Has<Lambda<CoreExpr>>());
  const auto& lambda = atomic.Get<Lambda<CoreExpr>>();
  CpsExprInterpreter<Val> interpreter{};
  DrrCtx<Val, Node<Val>> ctx{std::make_shared<graph::NodeArena<Node<Val>>>(),
                             std::nullopt,
                             std::nullopt,
                             std::nullopt};
  const auto& interpret_ret =
      interpreter.Interpret(lambda, std::vector<Val>{ctx});
  if (!interpret_ret.HasOkValue()) {
    LOG(ERROR) << interpret_ret.GetError().class_name() << ": "
               << interpret_ret.GetError().msg();
  }
  ASSERT_TRUE(interpret_ret.HasOkValue());
}

/*
python code:

def SoftmaxFusionDemo(ctx):
  ctx.pass_name = "softmax_prologue"

*/
TEST(DrrValue, setattr_pass_name) {
  const std::string json_str = R"(
    [
      "lambda",
      [
        "ctx"
      ],
      [
        "__builtin_let__",
        [
          [
            "__lambda_expr_tmp0",
            [
              "__builtin_setattr__",
              "ctx",
              {
                "str": "pass_name"
              }
            ]
          ],
          [
            "__lambda_expr_tmp1",
            [
              "__lambda_expr_tmp0",
              {
                "str": "pass_name"
              },
              {
                "str": "softmax_prologue"
              }
            ]
          ]
        ],
        [
          "__builtin_identity__",
          "__lambda_expr_tmp1"
        ]
      ]
    ]
  )";
  const auto& anf_expr = ap::axpr::MakeAnfExprFromJsonString(json_str);
  if (anf_expr.HasError()) {
    LOG(ERROR) << anf_expr.GetError().class_name() << ": "
               << anf_expr.GetError().msg();
  }
  ASSERT_TRUE(anf_expr.HasOkValue());
  const auto& core_expr = ConvertAnfExprToCoreExpr(anf_expr.GetOkValue());
  ASSERT_TRUE(core_expr.Has<Atomic<CoreExpr>>());
  const auto& atomic = core_expr.Get<Atomic<CoreExpr>>();
  ASSERT_TRUE(atomic.Has<Lambda<CoreExpr>>());
  const auto& lambda = atomic.Get<Lambda<CoreExpr>>();
  CpsExprInterpreter<Val> interpreter{};
  DrrCtx<Val, Node<Val>> ctx{std::make_shared<graph::NodeArena<Node<Val>>>(),
                             std::nullopt,
                             std::nullopt,
                             std::nullopt};
  const auto& interpret_ret =
      interpreter.Interpret(lambda, std::vector<Val>{ctx});
  if (!interpret_ret.HasOkValue()) {
    LOG(ERROR) << interpret_ret.GetError().class_name() << ": "
               << interpret_ret.GetError().msg();
  }
  ASSERT_TRUE(interpret_ret.HasOkValue());
}

/*
python code:

def SoftmaxFusionDemo(ctx):
  ctx.pass_name = "softmax_prologue"
  @ctx.source_pattern
  def SourcePattern(o, t):
    o.trivial_op = o.ap_trivial_fusion_op()
    o.trivial_op(
      [*t.inputs],
      [t.tensor0, *t.tensor0_siblings]
    )
    o.softmax_op = o.ap_native_op("pd_op.softmax")
    o.softmax_op(
      [t.tensor0],
      [t.tensor1]
    )

  @ctx.result_pattern
  def ResultPattern(o, t):
    o.fustion_op = o.ap_pattern_fusion_op()
    o.fustion_op(
      [*t.inputs],
      [t.tensor1, *t.tensor0_siblings]
    )
*/
TEST(DrrValue, demo) {
  const std::string json_str = R"(
    [
      "lambda",
      [
        "ctx"
      ],
      [
        "__builtin_let__",
        [
          [
            "__lambda_expr_tmp0",
            [
              "__builtin_setattr__",
              "ctx",
              {
                "str": "pass_name"
              }
            ]
          ],
          [
            "__lambda_expr_tmp1",
            [
              "__lambda_expr_tmp0",
              {
                "str": "pass_name"
              },
              {
                "str": "softmax_prologue"
              }
            ]
          ],
          [
            "__lambda_expr_tmp27",
            [
              "__builtin_getattr__",
              "ctx",
              {
                "str": "source_pattern"
              }
            ]
          ],
          [
            "__lambda_expr_tmp28",
            [
              "__lambda_expr_tmp27",
              [
                "lambda",
                [
                  "o",
                  "t"
                ],
                [
                  "__builtin_let__",
                  [
                    [
                      "__lambda_expr_tmp2",
                      [
                        "__builtin_getattr__",
                        "o",
                        {
                          "str": "ap_trivial_fusion_op"
                        }
                      ]
                    ],
                    [
                      "__lambda_expr_tmp3",
                      [
                        "__lambda_expr_tmp2"
                      ]
                    ],
                    [
                      "__lambda_expr_tmp4",
                      [
                        "__builtin_setattr__",
                        "o",
                        {
                          "str": "trivial_op"
                        }
                      ]
                    ],
                    [
                      "__lambda_expr_tmp5",
                      [
                        "__lambda_expr_tmp4",
                        {
                          "str": "trivial_op"
                        },
                        "__lambda_expr_tmp3"
                      ]
                    ],
                    [
                      "__lambda_expr_tmp6",
                      [
                        "__builtin_getattr__",
                        "o",
                        {
                          "str": "trivial_op"
                        }
                      ]
                    ],
                    [
                      "__lambda_expr_tmp7",
                      [
                        "__builtin_getattr__",
                        "t",
                        {
                          "str": "inputs"
                        }
                      ]
                    ],
                    [
                      "__lambda_expr_tmp8",
                      [
                        "__builtin_starred__",
                        "__lambda_expr_tmp7"
                      ]
                    ],
                    [
                      "__lambda_expr_tmp9",
                      [
                        "__builtin_list__",
                        "__lambda_expr_tmp8"
                      ]
                    ],
                    [
                      "__lambda_expr_tmp10",
                      [
                        "__builtin_getattr__",
                        "t",
                        {
                          "str": "tensor0"
                        }
                      ]
                    ],
                    [
                      "__lambda_expr_tmp11",
                      [
                        "__builtin_getattr__",
                        "t",
                        {
                          "str": "tensor0_siblings"
                        }
                      ]
                    ],
                    [
                      "__lambda_expr_tmp12",
                      [
                        "__builtin_starred__",
                        "__lambda_expr_tmp11"
                      ]
                    ],
                    [
                      "__lambda_expr_tmp13",
                      [
                        "__builtin_list__",
                        "__lambda_expr_tmp10",
                        "__lambda_expr_tmp12"
                      ]
                    ],
                    [
                      "__lambda_expr_tmp14",
                      [
                        "__lambda_expr_tmp6",
                        "__lambda_expr_tmp9",
                        "__lambda_expr_tmp13"
                      ]
                    ],
                    [
                      "__lambda_expr_tmp15",
                      [
                        "__builtin_identity__",
                        "__lambda_expr_tmp14"
                      ]
                    ],
                    [
                      "__lambda_expr_tmp16",
                      [
                        "__builtin_getattr__",
                        "o",
                        {
                          "str": "ap_native_op"
                        }
                      ]
                    ],
                    [
                      "__lambda_expr_tmp17",
                      [
                        "__lambda_expr_tmp16",
                        {
                          "str": "pd_op.softmax"
                        }
                      ]
                    ],
                    [
                      "__lambda_expr_tmp18",
                      [
                        "__builtin_setattr__",
                        "o",
                        {
                          "str": "softmax_op"
                        }
                      ]
                    ],
                    [
                      "__lambda_expr_tmp19",
                      [
                        "__lambda_expr_tmp18",
                        {
                          "str": "softmax_op"
                        },
                        "__lambda_expr_tmp17"
                      ]
                    ],
                    [
                      "__lambda_expr_tmp20",
                      [
                        "__builtin_getattr__",
                        "o",
                        {
                          "str": "softmax_op"
                        }
                      ]
                    ],
                    [
                      "__lambda_expr_tmp21",
                      [
                        "__builtin_getattr__",
                        "t",
                        {
                          "str": "tensor0"
                        }
                      ]
                    ],
                    [
                      "__lambda_expr_tmp22",
                      [
                        "__builtin_list__",
                        "__lambda_expr_tmp21"
                      ]
                    ],
                    [
                      "__lambda_expr_tmp23",
                      [
                        "__builtin_getattr__",
                        "t",
                        {
                          "str": "tensor1"
                        }
                      ]
                    ],
                    [
                      "__lambda_expr_tmp24",
                      [
                        "__builtin_list__",
                        "__lambda_expr_tmp23"
                      ]
                    ],
                    [
                      "__lambda_expr_tmp25",
                      [
                        "__lambda_expr_tmp20",
                        "__lambda_expr_tmp22",
                        "__lambda_expr_tmp24"
                      ]
                    ],
                    [
                      "__lambda_expr_tmp26",
                      [
                        "__builtin_identity__",
                        "__lambda_expr_tmp25"
                      ]
                    ]
                  ],
                  [
                    "__builtin_identity__",
                    "__lambda_expr_tmp26"
                  ]
                ]
              ]
            ]
          ],
          [
            "SourcePattern",
            [
              "__builtin_identity__",
              "__lambda_expr_tmp28"
            ]
          ],
          [
            "__lambda_expr_tmp43",
            [
              "__builtin_getattr__",
              "ctx",
              {
                "str": "result_pattern"
              }
            ]
          ],
          [
            "__lambda_expr_tmp44",
            [
              "__lambda_expr_tmp43",
              [
                "lambda",
                [
                  "o",
                  "t"
                ],
                [
                  "__builtin_let__",
                  [
                    [
                      "__lambda_expr_tmp29",
                      [
                        "__builtin_getattr__",
                        "o",
                        {
                          "str": "ap_pattern_fusion_op"
                        }
                      ]
                    ],
                    [
                      "__lambda_expr_tmp30",
                      [
                        "__lambda_expr_tmp29"
                      ]
                    ],
                    [
                      "__lambda_expr_tmp31",
                      [
                        "__builtin_setattr__",
                        "o",
                        {
                          "str": "fustion_op"
                        }
                      ]
                    ],
                    [
                      "__lambda_expr_tmp32",
                      [
                        "__lambda_expr_tmp31",
                        {
                          "str": "fustion_op"
                        },
                        "__lambda_expr_tmp30"
                      ]
                    ],
                    [
                      "__lambda_expr_tmp33",
                      [
                        "__builtin_getattr__",
                        "o",
                        {
                          "str": "fustion_op"
                        }
                      ]
                    ],
                    [
                      "__lambda_expr_tmp34",
                      [
                        "__builtin_getattr__",
                        "t",
                        {
                          "str": "inputs"
                        }
                      ]
                    ],
                    [
                      "__lambda_expr_tmp35",
                      [
                        "__builtin_starred__",
                        "__lambda_expr_tmp34"
                      ]
                    ],
                    [
                      "__lambda_expr_tmp36",
                      [
                        "__builtin_list__",
                        "__lambda_expr_tmp35"
                      ]
                    ],
                    [
                      "__lambda_expr_tmp37",
                      [
                        "__builtin_getattr__",
                        "t",
                        {
                          "str": "tensor1"
                        }
                      ]
                    ],
                    [
                      "__lambda_expr_tmp38",
                      [
                        "__builtin_getattr__",
                        "t",
                        {
                          "str": "tensor0_siblings"
                        }
                      ]
                    ],
                    [
                      "__lambda_expr_tmp39",
                      [
                        "__builtin_starred__",
                        "__lambda_expr_tmp38"
                      ]
                    ],
                    [
                      "__lambda_expr_tmp40",
                      [
                        "__builtin_list__",
                        "__lambda_expr_tmp37",
                        "__lambda_expr_tmp39"
                      ]
                    ],
                    [
                      "__lambda_expr_tmp41",
                      [
                        "__lambda_expr_tmp33",
                        "__lambda_expr_tmp36",
                        "__lambda_expr_tmp40"
                      ]
                    ],
                    [
                      "__lambda_expr_tmp42",
                      [
                        "__builtin_identity__",
                        "__lambda_expr_tmp41"
                      ]
                    ]
                  ],
                  [
                    "__builtin_identity__",
                    "__lambda_expr_tmp42"
                  ]
                ]
              ]
            ]
          ],
          [
            "ResultPattern",
            [
              "__builtin_identity__",
              "__lambda_expr_tmp44"
            ]
          ]
        ],
        [
          "__builtin_identity__",
          "ResultPattern"
        ]
      ]
    ]
  )";
  const auto& anf_expr = ap::axpr::MakeAnfExprFromJsonString(json_str);
  if (anf_expr.HasError()) {
    LOG(ERROR) << anf_expr.GetError().class_name() << ": "
               << anf_expr.GetError().msg();
  }
  ASSERT_TRUE(anf_expr.HasOkValue());
  const auto& core_expr = ConvertAnfExprToCoreExpr(anf_expr.GetOkValue());
  ASSERT_TRUE(core_expr.Has<Atomic<CoreExpr>>());
  const auto& atomic = core_expr.Get<Atomic<CoreExpr>>();
  ASSERT_TRUE(atomic.Has<Lambda<CoreExpr>>());
  const auto& lambda = atomic.Get<Lambda<CoreExpr>>();
  CpsExprInterpreter<Val> interpreter{};
  DrrCtx<Val, Node<Val>> ctx{std::make_shared<graph::NodeArena<Node<Val>>>(),
                             std::nullopt,
                             std::nullopt,
                             std::nullopt};
  const auto& interpret_ret =
      interpreter.Interpret(lambda, std::vector<Val>{ctx});
  if (!interpret_ret.HasOkValue()) {
    LOG(ERROR) << interpret_ret.GetError().class_name() << ": "
               << interpret_ret.GetError().msg();
  }
  ASSERT_TRUE(interpret_ret.HasOkValue());
}

}  // namespace ap::drr::tests
