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

// Laplace problem: Assembly using weak forms
// This test replicates step-6, but with a constant coefficient of unity.

#include <weak_forms/weak_forms.h>

#include "../weak_forms_tests.h"
#include "wf_common_tests/step-6.h"


using namespace dealii;


template <int dim>
class Step6 : public Step6_Base<dim>
{
public:
  Step6();

protected:
  void
  assemble_system() override;
};


template <int dim>
Step6<dim>::Step6()
  : Step6_Base<dim>()
{}


template <int dim>
void
Step6<dim>::assemble_system()
{
  using namespace WeakForms;
  constexpr int spacedim = dim;

  // Symbolic types for test function, trial solution and a coefficient.
  const TestFunction<dim>  test;
  const TrialSolution<dim> trial;
  const ScalarFunctor      mat_coeff("c", "c");
  const ScalarFunctor      rhs_coeff("s", "s");

  const auto test_val   = value(test);
  const auto test_grad  = gradient(test);
  const auto trial_grad = gradient(trial);
  const auto mat_coeff_func =
    value<double, dim, spacedim>(mat_coeff,
                                 [](const FEValuesBase<dim, spacedim> &,
                                    const unsigned int) { return 1.0; });
  const auto rhs_coeff_func =
    value<double, dim, spacedim>(rhs_coeff,
                                 [](const FEValuesBase<dim, spacedim> &,
                                    const unsigned int) { return 1.0; });

  MatrixBasedAssembler<dim> assembler;
  assembler += bilinear_form(test_grad, mat_coeff_func, trial_grad)
                 .dV();                                    // LHS contribution
  assembler -= linear_form(test_val, rhs_coeff_func).dV(); // RHS contribution

  // Other candidates for weak form types:
  // 1. assembler -= energy_form(psi(F,T,...)).dV();
  //  --> Generates: linear_form(dF, dpsi(F,T,...)/dF)
  //               + linear_form(dT, dpsi(F,T,...)/dT)
  //               + linear_form(...) // Residual contributions
  //               + bilinear_form(dF, d2psi(F,T,...)/dF.dF, DF)
  //               + bilinear_form(dF, d2psi(F,T,...)/dF.dT, DT)
  //               + bilinear_form(dT, d2psi(F,T,...)/dT.dF, DF)
  //               + bilinear_form(dT, d2psi(F,T,...)/dT.dT, DT)
  //               + bilinear_form(...) // Linearisations
  // 2. assembler -= residual_form(dF, P(F,T,...)).dV();
  //  --> Generates: linear_form(dF, P(F,T,...))
  //               + bilinear_form(dF, dP(F,T,...)/dF, FF)
  //               + bilinear_form(dF, dP(F,T,...)/dT, FT)
  //               + bilinear_form(dF, ...) // Linearisations

  // Look at what we're going to compute
  const SymbolicDecorations decorator;
  static bool               output = true;
  if (output)
    {
      std::cout << "Weak form (ascii):\n"
                << assembler.as_ascii(decorator) << std::endl;
      std::cout << "Weak form (LaTeX):\n"
                << assembler.as_latex(decorator) << std::endl;
      output = false;
    }

  // Now we pass in concrete objects to get data from
  // and assemble into.
  assembler.assemble_system(this->system_matrix,
                            this->system_rhs,
                            this->constraints,
                            this->dof_handler,
                            this->qf_cell);
}


int
main(int argc, char **argv)
{
  initlog();
  deallog << std::setprecision(9);

  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  try
    {
      Step6<2> laplace_problem_2d;
      laplace_problem_2d.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  deallog << "OK" << std::endl;

  return 0;
}
