file(GLOB_RECURSE axpr_srcs "src/axpr/*.cc")
set(axpr_deps common)

cc_library(
  axpr
  SRCS ${axpr_srcs}
  DEPS ${axpr_deps})

file(GLOB_RECURSE index_expr_srcs "src/index_expr/*.cc")
set(index_expr_deps axpr common)

cc_library(
  index_expr
  SRCS ${index_expr_srcs}
  DEPS ${index_expr_deps})
