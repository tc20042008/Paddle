cc_test(
  test_ap_kernel_definer
  SRCS test_ap_kernel_definer.cc
  DEPS phi common)
cc_test(
  test_ap_kernel_dispatch
  SRCS test_ap_kernel_dispatch.cc
  DEPS phi common)
