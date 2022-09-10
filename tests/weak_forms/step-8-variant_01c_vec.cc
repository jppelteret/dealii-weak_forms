// ---------------------------------------------------------------------
//
// Copyright (C) 2021 - 2022 by Jean-Paul Pelteret
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

// Elasticity problem: Assembly using weak forms
// This variant directly uses the scalar coefficients in the weak form.
// This test replicates step-8 exactly.
// - Vectorized variant

#include <deal.II/base/function.h>

#include <weak_forms/weak_forms.h>

#include "../weak_forms_tests.h"
#include "wf_common_tests/step-8.h"


using namespace dealii;



template <int dim>
class Step8 : public Step8_Base<dim>
{
public:
  Step8();

protected:
  void
  assemble_system() override;
};


template <int dim>
Step8<dim>::Step8()
  : Step8_Base<dim>()
{}


template <int dim>
void
Step8<dim>::assemble_system()
{
  using namespace WeakForms;

  // Symbolic types for test function, trial solution and a coefficient.
  const TestFunction<dim>          test;
  const TrialSolution<dim>         trial;
  const SubSpaceExtractors::Vector subspace_extractor(0, "u", "\\mathbf{u}");

  // Material coefficients
  const double lambda = 1.0;
  const double mu     = 1.0;

  const VectorFunctionFunctor<dim> rhs_coeff("s", "\\mathbf{s}");
  const RightHandSide<dim>         rhs;

  const auto test_ss  = test[subspace_extractor];
  const auto trial_ss = trial[subspace_extractor];

  const auto test_val   = test_ss.value();
  const auto test_grad  = test_ss.gradient();
  const auto trial_grad = trial_ss.gradient();


  MatrixBasedAssembler<dim> assembler;
  assembler +=
    bilinear_form(test_grad, lambda, trial_grad)
      .template component_filter<multiplicity_I | dof_I_component_i |
                                 multiplicity_J | dof_J_component_j>()
      .dV() +
    bilinear_form(test_grad, mu, trial_grad)
      .template component_filter<multiplicity_I | dof_I_component_j |
                                 multiplicity_J | dof_J_component_i>()
      .dV() +
    bilinear_form(test_grad, mu, trial_grad).delta_IJ().dV() -
    linear_form(test_val, rhs_coeff.value(rhs)).dV();

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
  const QGauss<dim> qf_cell(this->fe.degree + 1);
  assembler.assemble_system(this->system_matrix,
                            this->system_rhs,
                            this->constraints,
                            this->dof_handler,
                            qf_cell);
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
      Step8<2> elastic_problem_2d;
      elastic_problem_2d.run();
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
