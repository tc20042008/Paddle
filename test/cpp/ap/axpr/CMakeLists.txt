cc_test(
  test_axpr
  SRCS test_axpr.cc
  DEPS common)

cc_test(
  test_cps_expr_interpreter
  SRCS test_cps_expr_interpreter.cc
  DEPS common)

cc_test(
  test_index_expr
  SRCS test_index_expr.cc
  DEPS common)
