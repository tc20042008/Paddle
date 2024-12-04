file(GLOB_RECURSE axpr_srcs "src/axpr/*.cc")
set(axpr_deps pir common)

cc_library(
  axpr
  SRCS ${axpr_srcs}
  DEPS ${axpr_deps})
