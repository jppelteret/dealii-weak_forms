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

// Minimal surface problem: Assembly using composite weak forms
// This test replicates step-72 (unassisted formulation) exactly.
// - Non-vectorized variant

#include <weak_forms/weak_forms.h>

#include "../weak_forms_tests.h"
#include "wf_common_tests/step-72.h"

namespace Step72
{
  template <int dim>
  class Step72 : public Step72_Base<dim>
  {
  public:
    Step72()
      : Step72_Base<dim>()
    {}

  protected:
    void
    assemble_system() override;
  };


  template <int dim>
  void
  Step72<dim>::assemble_system()
  {
    using namespace WeakForms;
    constexpr int spacedim = dim;

    this->system_matrix = 0;
    this->system_rhs    = 0;

    // Symbolic types for test function, trial solution and a coefficient.
    const TestFunction<dim, spacedim>  test;
    const TrialSolution<dim, spacedim> trial;
    const FieldSolution<dim, spacedim> field_solution;
    const SubSpaceExtractors::Scalar   subspace_extractor_u(0, "u", "u");

    // Test function (subspaced)
    const auto grad_test_u = test[subspace_extractor_u].gradient();

    // Trial solution (subspaces)
    const auto grad_trial_u = trial[subspace_extractor_u].gradient();

    // Field solution
    const auto grad_u = field_solution[subspace_extractor_u].gradient();
    const auto coeff  = 1.0 / sqrt(1.0 + grad_u * grad_u);

    // Assembly
    MatrixBasedAssembler<dim, dim, double, false> assembler;
    assembler += bilinear_form(grad_test_u, coeff, grad_trial_u).dV();
    assembler -= bilinear_form(grad_test_u,
                               coeff * coeff * coeff, // pow(coeff, 3)
                               (grad_trial_u * grad_u) * grad_u)
                   .dV();
    assembler += linear_form(grad_test_u, coeff * grad_u).dV();

    // Look at what we're going to compute
    const SymbolicDecorations decorator;
    static bool               output = true;
    if (output)
      {
        std::cout << "\n\n" << std::endl;
        std::cout << "Weak form (ascii):\n"
                  << assembler.as_ascii(decorator) << std::endl;
        std::cout << "Weak form (LaTeX):\n"
                  << assembler.as_latex(decorator) << std::endl;
        std::cout << "\n\n" << std::endl;
        output = false;
      }

    // Now we pass in concrete objects to get data from
    // and assemble into.
    assembler.assemble_system(this->system_matrix,
                              this->system_rhs,
                              this->current_solution,
                              this->hanging_node_constraints,
                              this->dof_handler,
                              this->quadrature_formula);

    std::map<dealii::types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(this->dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                       this->system_matrix,
                                       this->newton_update,
                                       this->system_rhs);
  }
} // namespace Step72

int
main(int argc, char *argv[])
{
  initlog();
  deallog << std::setprecision(9);

  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, testing_max_num_threads());

      std::string prm_file;
      if (argc > 1)
        prm_file = argv[1];
      else
        prm_file = "parameters.prm";

      const Step72::Step72_Parameters parameters;

      Step72::Step72<2> minimal_surface_problem_2d;
      minimal_surface_problem_2d.run(parameters.tolerance);
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
  return 0;
}
