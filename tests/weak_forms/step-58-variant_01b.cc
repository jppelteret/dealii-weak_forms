// ---------------------------------------------------------------------
//
// Copyright (C) 2018 - 2021 by the deal.II authors
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

// ---------------------------------------------------------------------
//
// Copyright (C) 2022 by Jean-Paul Pelteret
//
// This file is part of the Weak forms for deal.II library.
//
// The Weak forms for deal.II library is free software; you can use it,
// redistribute it, and/or modify it under the terms of the GNU Lesser
// General Public License as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later
// version. The full text of the license can be found in the file LICENSE
// at the top level of the Weak forms for deal.II distribution.
//
// ---------------------------------------------------------------------

// Nonlinear Schroedinger equation: Assembly using composite weak forms.
// This test replicates step-58.


#include <weak_forms/weak_forms.h>

#include "../weak_forms_tests.h"
#include "wf_common_tests/step-58.h"


using namespace dealii;


template <int dim>
class Step58 : public Step58_Base<dim>
{
public:
  Step58();

protected:
  void
  assemble_matrices() override;
};


template <int dim>
Step58<dim>::Step58()
  : Step58_Base<dim>()
{}


template <int dim>
void
Step58<dim>::assemble_matrices()
{
  using namespace WeakForms;
  constexpr int spacedim = dim;

  // Symbolic types for test function, trial solution and a coefficient.
  const TestFunction<dim, spacedim>  test;
  const TrialSolution<dim, spacedim> trial;

  const Potential<dim>                  potential;
  const ScalarFunctionFunctor<spacedim> potential_function("Psi", "\\Psi");

  const auto i =
    WeakForms::constant_scalar<dim>(std::complex<double>{0, 1}, "i", "i");
  const auto ts =
    WeakForms::constant_scalar<dim>(this->time_step, "dt", "\\Delta t");

  // Cannot (meaningfully) vectorise a complex-valued problem
  // The VectorizedArray class only supports complex numbers
  // for a single lane.
  using MatrixBasedAssembler_t =
    MatrixBasedAssembler<dim, dim, std::complex<double>, false>;

  // Assembly: LHS
  MatrixBasedAssembler_t assembler_lhs;
  assembler_lhs +=
    bilinear_form(test.value(), -i, trial.value()).dV() +
    bilinear_form(test.gradient(), 0.25 * ts, trial.gradient()).dV() +
    bilinear_form(test.value(),
                  0.5 * ts *
                    potential_function.template value<double, dim>(potential),
                  trial.value())
      .dV();

  // Assembly: RHS
  MatrixBasedAssembler_t assembler_rhs;
  assembler_rhs +=
    bilinear_form(test.value(), -i, trial.value()).dV() -
    bilinear_form(test.gradient(), 0.25 * ts, trial.gradient()).dV() -
    bilinear_form(test.value(),
                  0.5 * ts *
                    potential_function.template value<double, dim>(potential),
                  trial.value())
      .dV();

  // Look at what we're going to compute
  const SymbolicDecorations decorator;
  static bool               output = true;
  if (output)
    {
      deallog << "\n" << std::endl;
      deallog << "LHS:\n" << std::endl;
      deallog << "Weak form (ascii):\n"
              << assembler_lhs.as_ascii(decorator) << std::endl;
      deallog << "Weak form (LaTeX):\n"
              << assembler_lhs.as_latex(decorator) << std::endl;
      deallog << "\n" << std::endl;

      deallog << "RHS:\n" << std::endl;
      deallog << "Weak form (ascii):\n"
              << assembler_rhs.as_ascii(decorator) << std::endl;
      deallog << "Weak form (LaTeX):\n"
              << assembler_rhs.as_latex(decorator) << std::endl;
      deallog << "\n" << std::endl;
      output = false;
    }

  // Now we pass in concrete objects to get data from
  // and assemble into.
  const QGauss<dim> cell_quadrature(this->fe.degree + 1);
  assembler_lhs.assemble_matrix(this->system_matrix,
                                this->constraints,
                                this->dof_handler,
                                cell_quadrature);
  assembler_rhs.assemble_matrix(this->rhs_matrix,
                                this->constraints,
                                this->dof_handler,
                                cell_quadrature);
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
      Step58<2> nse;
      nse.run();
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
