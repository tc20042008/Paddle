file(GLOB_RECURSE axpr_srcs "src/axpr/*.cc")
set(axpr_deps common)

cc_library(
  axpr
  SRCS ${axpr_srcs}
  DEPS ${axpr_deps})

file(GLOB_RECURSE index_expr_srcs "src/index_expr/*.cc")
set(index_expr_deps axpr)

cc_library(
  index_expr
  SRCS ${index_expr_srcs}
  DEPS ${index_expr_deps})

file(GLOB_RECURSE ap_drr_srcs "src/drr/*.cc")
set(ap_drr_deps axpr)

cc_library(
  ap_drr
  SRCS ${ap_drr_srcs}
  DEPS ${ap_drr_deps})

file(GLOB_RECURSE ap_phi_srcs "src/paddle/phi/*.cc")
set(ap_phi_deps axpr)

cc_library(
  ap_phi
  SRCS ${ap_phi_srcs}
  DEPS ${ap_phi_deps})

file(GLOB_RECURSE ap_pir_srcs "src/paddle/pir/*.cc")
set(ap_pir_deps axpr ap_drr)

cc_library(
  ap_pir
  SRCS ${ap_pir_srcs}
  DEPS ${ap_pir_deps})

file(GLOB_RECURSE ap_reified_drr_srcs "src/reified_drr/*.cc")
set(ap_reified_drr_deps axpr ap_drr)

cc_library(
  ap_reified_drr
  SRCS ${ap_reified_drr_srcs}
  DEPS ${ap_reified_drr_deps})

file(GLOB_RECURSE ap_cinn_srcs "src/paddle/cinn/*.cc")
set(ap_cinn_deps axpr ap_pir index_expr ap_drr ap_reified_drr)

cc_library(
  ap_cinn
  SRCS ${ap_cinn_srcs}
  DEPS ${ap_cinn_deps})
