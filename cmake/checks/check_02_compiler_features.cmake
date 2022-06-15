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


# Save the current flags
SET(CMAKE_REQUIRED_LIBRARIES_SAVED ${CMAKE_REQUIRED_LIBRARIES})
SET(CMAKE_REQUIRED_INCLUDES_SAVED  ${CMAKE_REQUIRED_INCLUDES})
SET(CMAKE_REQUIRED_FLAGS_SAVED     ${CMAKE_REQUIRED_FLAGS})

# Add deal.II's flags to the current project's
# (which should be empty unless the user specifies otherwise)
LIST(APPEND CMAKE_REQUIRED_LIBRARIES ${DEAL_II_LIBRARIES})
LIST(APPEND CMAKE_REQUIRED_INCLUDES  ${DEAL_II_INCLUDE_DIRS})
LIST(APPEND CMAKE_REQUIRED_FLAGS     ${DEAL_II_CXX_FLAGS})

IF("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  LIST(APPEND CMAKE_REQUIRED_FLAGS   ${DEAL_II_CXX_FLAGS_RELEASE})
ELSEIF("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  LIST(APPEND CMAKE_REQUIRED_FLAGS   ${DEAL_II_CXX_FLAGS_DEBUG})
ELSE()
  MESSAGE(FATAL_ERROR "CMAKE_BUILD_TYPE doesn't match either Release or Debug!")
ENDIF()

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
  class BinaryOp
  {
  public:
    template <typename T>
    T
    operator()(const T &num, const T &den) const
    {
      return num/den;
    }
  };
  template <typename Number, std::size_t width>
  void
  do_test()
  {
    using Vec_t = VectorizedArray<Number, width>;
    for (unsigned int i=0; i<10000; ++i)
    {
      const Vec_t num = 1.0;
      const Vec_t den = 0.0;
      auto result = num / den;
      result = BinaryOp()(num, den);
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
# any way sporadic. The tested value 6.916...e-310 comes from some
# backtraced output of a test failure on a Docker image. 
#
WF_CHECK_CXX_SOURCE_RUNS(
  "
  #include <deal.II/base/vectorization.h>
  #include <limits>
  using namespace dealii;
  class UnaryOp
  {
  public:
    template <typename T>
    T
    operator()(const T &value) const
    {
      using namespace std;
      return sqrt(value);
    }
  };
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
      result = UnaryOp()(val);
      val = std::numeric_limits<Number>::epsilon();
      result = sqrt(val);
      result = UnaryOp()(val);
      val = 6.9161116785120245e-310;
      result = sqrt(val);
      result = UnaryOp()(val);
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
    
    // All indications are that _mm_sqrt_pd() sporadically segfaults with zero input vector.
  #if DEAL_II_VECTORIZATION_WIDTH_IN_BITS == 128
    static_assert(false, 'Problematic vectorization width detected.');
  #endif

    do_test<double, 1>();
    do_test<float, 1>();
    
    return 0;
  }
  "
  WEAK_FORMS_VECTORIZATION_FPE_SQRT_OF_ZERO)


# Restore all flags
SET(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES_SAVED})
SET(CMAKE_REQUIRED_INCLUDES  ${CMAKE_REQUIRED_INCLUDES_SAVED})
SET(CMAKE_REQUIRED_FLAGS     ${CMAKE_REQUIRED_FLAGS_SAVED})
