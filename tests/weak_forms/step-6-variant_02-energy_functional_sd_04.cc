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
// in conjunction with symbolic differentiation.
// The energy functional form is recovered directly from a symbolic function.
// This test replicates step-6 exactly.
// - Optimizer type: LLVM
// - Optimization method: All

#include <deal.II/differentiation/sd.h>

#include <weak_forms/weak_forms.h>

#include "../weak_forms_tests.h"
#include "wf_common_tests/step-6.h"

WEAK_FORMS_NAMESPACE_OPEN

namespace WeakForms
{
  namespace Operators
  {
    namespace internal
    {
      template <typename T, typename U = void>
      struct CompositeOpHelper;

      // Symbolic op (field op)
      template <typename T>
      struct CompositeOpHelper<
        T,
        typename std::enable_if<is_subspace_field_solution_op<T>::value &&
                                !is_unary_op<T>::value &&
                                !is_binary_op<T>::value>::type>
      {
        static void
        print(const T &op)
        {
          std::cout << "op (field solution): "
                    << op.as_ascii(SymbolicDecorations()) << std::endl;
        }

        static std::tuple<T>
        get_subspace_field_solution_ops(const T &op)
        {
          return std::make_tuple(op);
        }
      };

      // Symbolic op (not a field op)
      template <typename T>
      struct CompositeOpHelper<
        T,
        typename std::enable_if<!is_subspace_field_solution_op<T>::value &&
                                !is_unary_op<T>::value &&
                                !is_binary_op<T>::value>::type>
      {
        static void
        print(const T &op)
        {
          std::cout << "op (not field solution): "
                    << op.as_ascii(SymbolicDecorations()) << std::endl;
        }

        static std::tuple<>
        get_subspace_field_solution_ops(const T &op)
        {
          // An empty tuple
          return std::make_tuple();
        }
      };

      // Unary op
      template <typename T>
      struct CompositeOpHelper<
        T,
        typename std::enable_if<is_unary_op<T>::value>::type>
      {
        static void
        print(const T &op)
        {
          std::cout << "unary op" << std::endl;

          using OpType = typename T::OpType;
          CompositeOpHelper<OpType>::print(op.get_operand());
        }

        static auto
        get_subspace_field_solution_ops(const T &op)
        {
          using OpType = typename T::OpType;
          return CompositeOpHelper<OpType>::get_subspace_field_solution_ops(
            op.get_operand());
        }
      };

      // Binary op
      template <typename T>
      struct CompositeOpHelper<
        T,
        typename std::enable_if<is_binary_op<T>::value>::type>
      {
        static void
        print(const T &op)
        {
          std::cout << "binary op" << std::endl;

          using LhsOpType = typename T::LhsOpType;
          using RhsOpType = typename T::RhsOpType;
          CompositeOpHelper<LhsOpType>::print(op.get_lhs_operand());
          CompositeOpHelper<RhsOpType>::print(op.get_rhs_operand());
        }

        static auto
        get_subspace_field_solution_ops(const T &op)
        {
          using LhsOpType = typename T::LhsOpType;
          using RhsOpType = typename T::RhsOpType;

          return std::tuple_cat(
            CompositeOpHelper<LhsOpType>::get_subspace_field_solution_ops(
              op.get_lhs_operand()),
            CompositeOpHelper<RhsOpType>::get_subspace_field_solution_ops(
              op.get_rhs_operand()));
        }
      };

    } // namespace internal
  }   // namespace Operators
} // namespace WeakForms

WEAK_FORMS_NAMESPACE_CLOSE


using namespace dealii;


namespace dealiiWF = dealiiWeakForms::WeakForms;


template <typename... SymbolicOpsSubSpaceFieldSolution>
auto
create_energy_functor_from_tuple(
  const std::tuple<SymbolicOpsSubSpaceFieldSolution...>
    &subspace_field_solution_ops)
{
  return dealiiWF::EnergyFunctor<SymbolicOpsSubSpaceFieldSolution...>(
    "e", "\\Psi", subspace_field_solution_ops);
}


template <int dim,
          int spacedim = dim,
          typename CompositeSymbolicOp,
          typename... SymbolicOpsSubSpaceFieldSolution>
auto
create_energy_functional_form_from_energy(
  const dealiiWF::EnergyFunctor<SymbolicOpsSubSpaceFieldSolution...>
    &                        energy_functor,
  const CompositeSymbolicOp &functor_op)
{
  static_assert(
    CompositeSymbolicOp::rank == 0,
    "Expect functor for energy functional form to return a scalar upon evaluation.");

  using SDNumberType = Differentiation::SD::Expression;
  using EnergyFunctorType =
    dealiiWF::EnergyFunctor<SymbolicOpsSubSpaceFieldSolution...>;
  using substitution_map_type =
    typename EnergyFunctorType::substitution_map_type;

  const auto energy =
    energy_functor.template value<SDNumberType, dim, spacedim>(
      [functor_op](
        const typename SymbolicOpsSubSpaceFieldSolution::template value_type<
          SDNumberType> &...field_solutions)
      {
        // The expression is filled with the full scalar expression as is
        // returned by the user-defined functor. The symbols used for the field
        // solution operations are consistent with that which fills the argument
        // list, and we therefore can be assured that they will be given the
        // correct value upon later substitution.
        return functor_op.as_expression();
      },
      [functor_op](
        const typename SymbolicOpsSubSpaceFieldSolution::template value_type<
          SDNumberType> &...field_solutions)
      {
        // Extract from functor_op...
        // We really only expect user-defined symbolic variables to be
        // able to return an intermediate substitution value.

        // return {Differentiation::SD::make_symbol_map(coefficient)};
        return functor_op.get_intermediate_substitution_map();
      },
      [functor_op](
        const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
        const std::vector<dealiiWF::SolutionExtractionData<dim, spacedim>>
          &                solution_extraction_data,
        const unsigned int q_point)
      {
        // Extract from functor_op...
        // Here we get the point-specific values from all of the variables used
        // in the expression tree (i.e. the functor). The exception to this are
        // the field solution variables, which have their values written into
        // the substitution map by the framework.

        // return Differentiation::SD::make_substitution_map(coefficient, c);
        return functor_op.get_substitution_map(scratch_data,
                                               solution_extraction_data,
                                               q_point);
      },
      Differentiation::SD::OptimizerType::llvm,
      Differentiation::SD::OptimizationFlags::optimize_default,
      functor_op.get_update_flags());

  return dealiiWF::energy_functional_form(energy);
}



template <int dim, int spacedim = dim, typename CompositeSymbolicOp>
auto
energy_functional_form(const CompositeSymbolicOp &functor_op)
{
  using namespace dealiiWF::Operators::internal;

  // A tuple of all field solution operations that appear in the symbolic
  // expression tree. Some of these are duplicated: that's fine, as they'll
  // be a no-op when producing the symbolic substitution maps.
  const auto subspace_field_solution_ops =
    CompositeOpHelper<CompositeSymbolicOp>::get_subspace_field_solution_ops(
      functor_op);

  // Create the energy functor, separately from the energy functional form.
  // We do this because we need to explicitly know the template parameters that
  // are passed to both the energy functor and the energy functional form
  // creation methods.
  const auto energy_func =
    create_energy_functor_from_tuple(subspace_field_solution_ops);

  // Finally create the energy functional form. The original functor is needed
  // in order to extract the substitution map for the variables other than the
  // field solution components.
  return create_energy_functional_form_from_energy<dim, spacedim>(energy_func,
                                                                  functor_op);
}


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

  constexpr int spacedim = dim;
  using SDNumber_t       = Differentiation::SD::Expression;

  // Symbolic types for test function, trial solution and a coefficient.
  const TestFunction<dim>  test;
  const FieldSolution<dim> solution;
  const ScalarFunctor      rhs_coeff("s", "s");

  const WeakForms::SubSpaceExtractors::Scalar subspace_extractor(0, "s", "s");

  const auto test_ss  = test[subspace_extractor];
  const auto test_val = test_ss.value();

  const auto soln_ss   = solution[subspace_extractor];
  const auto soln_grad = soln_ss.gradient();

  const auto coefficient = ScalarFunctor("c1", "c1")
                             .template value<double, dim, spacedim>(
                               [](const FEValuesBase<dim, spacedim> &fe_values,
                                  const unsigned int                 q_point)
                               {
                                 const Point<spacedim> &p =
                                   fe_values.get_quadrature_points()[q_point];
                                 return (p.square() < 0.5 * 0.5 ? 20.0 : 1.0);
                               });

  const auto energy = 0.5 * coefficient * scalar_product(soln_grad, soln_grad);

  const auto rhs_coeff_func = rhs_coeff.template value<double, dim, spacedim>(
    [](const FEValuesBase<dim, spacedim> &, const unsigned int)
    { return 1.0; });

  MatrixBasedAssembler<dim> assembler;
  assembler += energy_functional_form<dim, spacedim>(energy).dV();
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
