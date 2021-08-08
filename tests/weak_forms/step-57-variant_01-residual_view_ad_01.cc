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

// Incompressible Navier-Stokes flow problem: Assembly using self-linearizing
// residual weak form in conjunction with automatic differentiation.
// This test replicates step-57 exactly.

#include <weak_forms/weak_forms.h>

#include "../weak_forms_tests.h"
#include "wf_common_tests/step-57.h"

namespace Step57
{
  template <int dim>
  class Step57 : public StationaryNavierStokes<dim>
  {
  public:
    Step57(const unsigned int degree)
      : StationaryNavierStokes<dim>(degree)
    {}

  protected:
    void
    assemble(const bool initial_step, const bool assemble_matrix) override;
  };


  template <int dim>
  void
  Step57<dim>::assemble(const bool initial_step, const bool assemble_matrix)
  {
    if (assemble_matrix)
      this->system_matrix = 0;

    this->system_rhs = 0;

    using namespace WeakForms;
    using namespace Differentiation;

    constexpr int  spacedim    = dim;
    constexpr auto ad_typecode = Differentiation::AD::NumberTypes::sacado_dfad;
    using ADNumber_t =
      typename Differentiation::AD::NumberTraits<double, ad_typecode>::ad_type;

    // Symbolic types for test function, trial solution and a coefficient.
    const TestFunction<dim, spacedim>  test;
    const TrialSolution<dim, spacedim> trial;
    const FieldSolution<dim, spacedim> field_solution;
    const SubSpaceExtractors::Vector   subspace_extractor_v(0,
                                                          "v",
                                                          "\\mathbf{v}");
    const SubSpaceExtractors::Scalar   subspace_extractor_p(spacedim,
                                                          "p_tilde",
                                                          "\\tilde{p}");


    // Test function (subspaces)
    const auto test_ss_v = test[subspace_extractor_v];
    const auto test_ss_p = test[subspace_extractor_p];

    const auto test_v      = test[subspace_extractor_v].value();
    const auto div_test_v  = test[subspace_extractor_v].divergence();
    const auto grad_test_v = test[subspace_extractor_v].gradient();
    const auto test_p      = test[subspace_extractor_p].value();

    // Trial solution (subspaces)
    const auto trial_p = trial[subspace_extractor_p].value();

    // Field solution
    const auto v      = field_solution[subspace_extractor_v].value();
    const auto div_v  = field_solution[subspace_extractor_v].divergence();
    const auto grad_v = field_solution[subspace_extractor_v].gradient();
    const auto p      = field_solution[subspace_extractor_p].value();

    // Residual
    const auto residual_func = residual_functor("R", "R", v, div_v, grad_v, p);
    const auto residual_ss_v = residual_func[test_v];
    const auto residual_ss_div_v  = residual_func[div_test_v];
    const auto residual_ss_grad_v = residual_func[grad_test_v];
    const auto residual_ss_p      = residual_func[test_p];

    const auto residual_v =
      residual_ss_v.template value<ADNumber_t, dim, spacedim>(
        [this,
         &spacedim](const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                    const std::vector<std::string> &       solution_names,
                    const unsigned int                     q_point,
                    const Tensor<1, spacedim, ADNumber_t> &v,
                    const ADNumber_t &                     div_v,
                    const Tensor<2, spacedim, ADNumber_t> &grad_v,
                    const ADNumber_t &                     p)
        {
          // Sacado is unbelievably annoying. If we don't explicitly
          // cast this return type then we get a segfault.
          // i.e. don't return the result inline!
          const Tensor<1, spacedim, ADNumber_t> res = -grad_v * v;
          return res;
        });

    const auto residual_div_v =
      residual_ss_div_v.template value<ADNumber_t, dim, spacedim>(
        [this,
         &spacedim](const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                    const std::vector<std::string> &       solution_names,
                    const unsigned int                     q_point,
                    const Tensor<1, spacedim, ADNumber_t> &v,
                    const ADNumber_t &                     div_v,
                    const Tensor<2, spacedim, ADNumber_t> &grad_v,
                    const ADNumber_t &                     p)
        {
          // Sacado is unbelievably annoying. If we don't explicitly
          // cast this return type then we get a segfault.
          // i.e. don't return the result inline!
          const ADNumber_t res = p - this->gamma * div_v;
          return res;
        });

    const auto residual_grad_v =
      residual_ss_grad_v.template value<ADNumber_t, dim, spacedim>(
        [this,
         &spacedim](const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                    const std::vector<std::string> &       solution_names,
                    const unsigned int                     q_point,
                    const Tensor<1, spacedim, ADNumber_t> &v,
                    const ADNumber_t &                     div_v,
                    const Tensor<2, spacedim, ADNumber_t> &grad_v,
                    const ADNumber_t &                     p)
        {
          // Sacado is unbelievably annoying. If we don't explicitly
          // cast this return type then we get a segfault.
          // i.e. don't return the result inline!
          const Tensor<2, spacedim, ADNumber_t> res = -this->viscosity * grad_v;
          return res;
        });

    const auto residual_p =
      residual_ss_p.template value<ADNumber_t, dim, spacedim>(
        [this,
         &spacedim](const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                    const std::vector<std::string> &       solution_names,
                    const unsigned int                     q_point,
                    const Tensor<1, spacedim, ADNumber_t> &v,
                    const ADNumber_t &                     div_v,
                    const Tensor<2, spacedim, ADNumber_t> &grad_v,
                    const ADNumber_t &                     p)
        {
          // Sacado is unbelievably annoying. If we don't explicitly
          // cast this return type then we get a segfault.
          // i.e. don't return the result inline!
          const ADNumber_t res = div_v;
          return res;
        });

    // Assembly
    MatrixBasedAssembler<dim> assembler;

    assembler +=
      residual_form(residual_v).dV() + residual_form(residual_div_v).dV() +
      residual_form(residual_grad_v).dV() + residual_form(residual_p).dV();

    // Additional mass-matrix terms for preconditioner
    assembler += bilinear_form(test_p, 1, trial_p).dV();

    // Look at what we're going to compute
    const SymbolicDecorations decorator;
    static bool               output = true;
    if (output)
      {
        deallog << "\n" << std::endl;
        deallog << "Weak form (ascii):\n"
                << assembler.as_ascii(decorator) << std::endl;
        deallog << "Weak form (LaTeX):\n"
                << assembler.as_latex(decorator) << std::endl;
        deallog << "\n" << std::endl;
        output = false;
      }

    // Now we pass in concrete objects to get data from
    // and assemble into.
    const AffineConstraints<double> &constraints_used =
      initial_step ? this->nonzero_constraints : this->zero_constraints;
    const QGauss<dim> quadrature_formula(this->degree + 2);
    const auto &      solution_vector = this->evaluation_point;
    if (assemble_matrix)
      {
        assembler.assemble_system(this->system_matrix,
                                  this->system_rhs,
                                  solution_vector,
                                  constraints_used,
                                  this->dof_handler,
                                  quadrature_formula);

        this->pressure_mass_matrix.reinit(this->sparsity_pattern.block(1, 1));
        this->pressure_mass_matrix.copy_from(this->system_matrix.block(1, 1));

        this->system_matrix.block(1, 1) = 0;
      }
    else
      {
        assembler.assemble_rhs_vector(this->system_rhs,
                                      solution_vector,
                                      constraints_used,
                                      this->dof_handler,
                                      quadrature_formula);
      }
  }

} // namespace Step57

int
main(int argc, char **argv)
{
  initlog();
  deallog << std::setprecision(9);

  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  using namespace dealii;
  try
    {
      const unsigned int dim                       = 2;
      const unsigned int degree                    = 1;
      const unsigned int n_local_refinement_levels = 1;

      Step57::Step57<dim> flow(degree);
      flow.run(n_local_refinement_levels);
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
