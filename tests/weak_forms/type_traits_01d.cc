// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------


// Check type traits for spaces
// - Sub-space: SymmetricTensor


#include <weak_forms/spaces.h>
#include <weak_forms/subspace_extractors.h>
#include <weak_forms/subspace_views.h>
#include <weak_forms/type_traits.h>

#include "../weak_forms_tests.h"

// https://stackoverflow.com/a/39742502
template <class T>
std::string
type_name()
{
#ifdef __clang__
  std::string p = __PRETTY_FUNCTION__;
  return p.substr(43, p.length() - 43 - 1);
#elif defined(__GNUC__)
  std::string p = __PRETTY_FUNCTION__;
#  if __cplusplus < 201402
  return p.substr(57, p.length() - 53 - 62);
#  else
  return p.substr(46, p.length() - 46 - 1);
#  endif
#elif defined(_MSC_VER)
  std::string p = __FUNCSIG__;
  return p.substr(38, p.length() - 38 - 7);
#else
  return std::string("This function is not supported!");
#endif
}


int
main()
{
  using namespace WeakForms;

  initlog();

  constexpr int dim      = 2;
  constexpr int spacedim = 2;
  constexpr int rank     = 2;

  const WeakForms::SubSpaceExtractors::SymmetricTensor<rank> subspace_extractor(
    0, "S", "\\mathbf{S}");

  using test_t  = TestFunction<dim, spacedim>;
  using trial_t = TrialSolution<dim, spacedim>;
  using soln_t  = FieldSolution<dim, spacedim>;

  using test_ss_t  = typename std::decay<decltype(
    std::declval<test_t>()[subspace_extractor])>::type;
  using trial_ss_t = typename std::decay<decltype(
    std::declval<trial_t>()[subspace_extractor])>::type;
  using soln_ss_t  = typename std::decay<decltype(
    std::declval<soln_t>()[subspace_extractor])>::type;

  // Print types
  std::cout << "test_t:     " << type_name<test_t>() << std::endl;
  std::cout << "test_ss_t:  " << type_name<test_ss_t>() << std::endl;
  std::cout << "trial_t:    " << type_name<trial_t>() << std::endl;
  std::cout << "trial_ss_t: " << type_name<trial_ss_t>() << std::endl;
  std::cout << "soln_t:     " << type_name<soln_t>() << std::endl;
  std::cout << "soln_ss_t:  " << type_name<soln_ss_t>() << std::endl
            << std::endl;

  deallog << std::boolalpha;

  deallog << "is_subspace_view()" << std::endl;
  deallog << is_subspace_view<test_ss_t>::value << std::endl;
  deallog << is_subspace_view<trial_ss_t>::value << std::endl;
  deallog << is_subspace_view<soln_ss_t>::value << std::endl;

  deallog << std::endl;

  deallog << "is_test_function()" << std::endl;
  deallog << is_test_function<test_ss_t>::value << std::endl;
  deallog << is_test_function<trial_ss_t>::value << std::endl;
  deallog << is_test_function<soln_ss_t>::value << std::endl;

  deallog << std::endl;

  deallog << "is_trial_solution()" << std::endl;
  deallog << is_trial_solution<test_ss_t>::value << std::endl;
  deallog << is_trial_solution<trial_ss_t>::value << std::endl;
  deallog << is_trial_solution<soln_ss_t>::value << std::endl;

  deallog << std::endl;

  deallog << "is_field_solution()" << std::endl;
  deallog << is_field_solution<test_ss_t>::value << std::endl;
  deallog << is_field_solution<trial_ss_t>::value << std::endl;
  deallog << is_field_solution<soln_ss_t>::value << std::endl;

  deallog << std::endl;

  deallog << "OK" << std::endl;
}
