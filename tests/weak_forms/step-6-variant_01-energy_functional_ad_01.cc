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

// Laplace problem: Assembly using self-linearizing energy functional weak form
// in conjunction with automatic differentiation. This test replicates step-6,
// but with a constant coefficient of unity.

#include <deal.II/differentiation/ad.h>

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
  using namespace Differentiation;

  constexpr int  spacedim = dim;
  constexpr auto ad_typecode =
    Differentiation::AD::NumberTypes::sacado_dfad_dfad;
  using ADNumber_t =
    typename Differentiation::AD::NumberTraits<double, ad_typecode>::ad_type;

  // Symbolic types for test function, trial solution and a coefficient.
  const TestFunction<dim>  test;
  const FieldSolution<dim> solution;
  const ScalarFunctor      rhs_coeff("s", "s");

  const WeakForms::SubSpaceExtractors::Scalar subspace_extractor(0, "s", "s");

  const auto test_ss  = test[subspace_extractor];
  const auto test_val = test_ss.value();

  const auto soln_ss   = solution[subspace_extractor];
  const auto soln_grad = soln_ss.gradient(); // Solution gradient

  const auto energy_func = energy_functor("e", "\\Psi", soln_grad);
  using EnergyADNumber_t =
    typename decltype(energy_func)::template ad_type<double, ad_typecode>;
  static_assert(std::is_same<ADNumber_t, EnergyADNumber_t>::value,
                "Expected identical AD number types");

  const auto energy = energy_func.template value<ADNumber_t, dim, spacedim>(
    [](const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
       const std::vector<SolutionExtractionData<dim, spacedim>>
         &                                    solution_extraction_data,
       const unsigned int                     q_point,
       const Tensor<1, spacedim, ADNumber_t> &grad_u)
    {
      // Sacado is unbelievably annoying. If we don't explicitly
      // cast this return type then we get a segfault.
      // i.e. don't return the result inline!
      const ADNumber_t energy = 0.5 * scalar_product(grad_u, grad_u);
      return energy;
    });

  const auto rhs_coeff_func = rhs_coeff.template value<double, dim, spacedim>(
    [](const FEValuesBase<dim, spacedim> &, const unsigned int)
    { return 1.0; });


  MatrixBasedAssembler<dim> assembler;
  assembler += energy_functional_form(energy).dV();
  assembler -= linear_form(test_val, rhs_coeff_func).dV(); // RHS contribution

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
  assembler.assemble_system(this->system_matrix,
                            this->system_rhs,
                            this->solution,
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
