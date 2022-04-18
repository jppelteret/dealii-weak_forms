## ---------------------------------------------------------------------
##
## Copyright (C) 2021 - 2022 by Jean-Paul Pelteret
##
## This file is part of the Weak forms for deal.II library.
##
## The Weak forms for deal.II library is free software; you can use it,
## redistribute it, and/or modify it under the terms of the GNU Lesser
## General Public License as published by the Free Software Foundation;
## either version 3.0 of the License, or (at your option) any later
## version. The full text of the license can be found in the file LICENSE
## at the top level of the Weak forms for deal.II distribution.
##
## ---------------------------------------------------------------------


LIST(APPEND CMAKE_REQUIRED_LIBRARIES ${DEAL_II_LIBRARIES})
LIST(APPEND CMAKE_REQUIRED_INCLUDES  ${DEAL_II_INCLUDE_DIRS})
LIST(APPEND CMAKE_REQUIRED_FLAGS     ${DEAL_II_CXX_FLAGS})

#
# Check whether division by zero in a vectorized array results in
# a floating point exception.
#
# The test is multiple times to ensure that the failure is not in
# any way sporadic.
#
WF_CHECK_CXX_SOURCE_RUNS(
  "
  #include <deal.II/base/vectorization.h>
  using namespace dealii;
  template <typename Number, std::size_t width>
  void
  do_test()
  {
    using Vec_t = VectorizedArray<Number, width>;
    for (unsigned int i=0; i<10000; ++i)
    {
      const Vec_t num = 1.0;
      const Vec_t den = 0.0;
      const auto result = num / den;
      (void)result;
    }
  }
  int main()
  {
  #if DEAL_II_VECTORIZATION_WIDTH_IN_BITS >= 512
    do_test<double, 8>();
    do_test<float, 16>();
  #endif

  #if DEAL_II_VECTORIZATION_WIDTH_IN_BITS >= 256
    do_test<double, 4>();
    do_test<float, 8>();
  #endif

  #if DEAL_II_VECTORIZATION_WIDTH_IN_BITS >= 128
    do_test<double, 2>();
    do_test<float, 4>();
  #endif

    do_test<double, 1>();
    do_test<float, 1>();
    
    return 0;
  }
  "
  WEAK_FORMS_VECTORIZATION_FPE_DIVIDE_BY_ZERO)


#
# Check whether taking the square root of a zero vectorized array
# results in a floating point exception.
#
# The test is multiple times to ensure that the failure is not in
# any way sporadic.
#
WF_CHECK_CXX_SOURCE_RUNS(
  "
  #include <deal.II/base/vectorization.h>
  #include <limits>
  using namespace dealii;
  template <typename Number, std::size_t width>
  void
  do_test()
  {
    using namespace std;
    using Vec_t = VectorizedArray<Number, width>;
    for (unsigned int i=0; i<10000; ++i)
    {
      Vec_t val = 0.0;
      auto result = sqrt(val);
      val = std::numeric_limits<Number>::epsilon();
      result = sqrt(val);
      (void)result;
    }
  }
  int main()
  {
  #if DEAL_II_VECTORIZATION_WIDTH_IN_BITS >= 512
    do_test<double, 8>();
    do_test<float, 16>();
  #endif

  #if DEAL_II_VECTORIZATION_WIDTH_IN_BITS >= 256
    do_test<double, 4>();
    do_test<float, 8>();
  #endif

  #if DEAL_II_VECTORIZATION_WIDTH_IN_BITS >= 128
    do_test<double, 2>();
    do_test<float, 4>();
  #endif

    do_test<double, 1>();
    do_test<float, 1>();
    
    return 0;
  }
  "
  WEAK_FORMS_VECTORIZATION_FPE_SQRT_OF_ZERO)