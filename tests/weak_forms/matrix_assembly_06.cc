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


// Check assembly of a matrix over an entire triangulation
// - Laplacian-Laplacian matrix (scalar-valued finite element)

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/meshworker/scratch_data.h>

#include <weak_forms/assembler_matrix_based.h>
#include <weak_forms/bilinear_forms.h>
#include <weak_forms/functors.h>
#include <weak_forms/spaces.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/symbolic_operators.h>

#include "../weak_forms_tests.h"
#include "wf_common_tests/utilities.h"


template <int dim, int spacedim = dim>
void
run()
{
  LogStream::Prefix prefix("Dim " + Utilities::to_string(dim));
  std::cout << "Dim: " << dim << std::endl;

  const FE_Q<dim, spacedim>  fe(2);
  const QGauss<spacedim>     qf_cell(fe.degree + 1);
  const QGauss<spacedim - 1> qf_face(fe.degree + 1);

  Triangulation<dim, spacedim> triangulation;
  GridGenerator::subdivided_hyper_cube(triangulation, 4, 0.0, 1.0);

  DoFHandler<dim, spacedim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix_std;
  SparseMatrix<double> system_matrix_wf;

  const UpdateFlags update_flags = update_hessians | update_JxW_values;

  {
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);

    sparsity_pattern.copy_from(dsp);

    system_matrix_std.reinit(sparsity_pattern);
    system_matrix_wf.reinit(sparsity_pattern);
  }

  auto verify_assembly = [](const SparseMatrix<double> &system_matrix_std,
                            const SparseMatrix<double> &system_matrix_wf)
  {
    constexpr double tol = 1e-12;

    for (auto it1 = system_matrix_std.begin(), it2 = system_matrix_wf.begin();
         it1 != system_matrix_std.end();
         ++it1, ++it2)
      {
        Assert(it2 != system_matrix_wf.end(), ExcInternalError());

        Assert(it1->row() == it2->row(),
               ExcIteratorRowIndexNotEqual(it1->row(), it2->row()));
        Assert(it1->column() == it2->column(),
               ExcIteratorColumnIndexNotEqual(it1->column(), it2->column()));

        AssertThrow(std::abs(it1->value() - it2->value()) < tol,
                    ExcMatrixEntriesNotEqual(
                      it1->row(), it1->column(), it1->value(), it2->value()));
      }
  };

  {
    std::cout << "Standard assembly" << std::endl;
    system_matrix_std = 0;

    FEValues<dim, spacedim> fe_values(fe, qf_cell, update_flags);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        fe_values.reinit(cell);

        for (const unsigned int q : fe_values.quadrature_point_indices())
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) += trace(fe_values.shape_hessian(i, q)) *
                                   trace(fe_values.shape_hessian(j, q)) *
                                   fe_values.JxW(q);


        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix,
                                               local_dof_indices,
                                               system_matrix_std);
      }

    // system_matrix_std.print(std::cout);
  }

  // Scalar valued coefficient
  {
    using namespace WeakForms;

    deallog << "Weak form assembly (bilinear form, scalar valued coefficient)"
            << std::endl;
    system_matrix_wf = 0;

    // Symbolic types for test function, trial solution and a coefficient.
    const TestFunction<dim, spacedim>  test;
    const TrialSolution<dim, spacedim> trial;
    const ScalarFunctor                coeff("c", "c");

    const auto test_lap   = test.laplacian();
    const auto trial_lap  = trial.laplacian();
    const auto coeff_func = coeff.template value<double, dim, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      { return 1.0; });

    // Still no concrete definitions
    MatrixBasedAssembler<dim, spacedim> assembler;
    assembler += bilinear_form(test_lap, coeff_func, trial_lap).dV();

    // Look at what we're going to compute
    const SymbolicDecorations decorator;
    deallog << "Weak form (ascii):\n"
            << assembler.as_ascii(decorator) << std::endl;
    deallog << "Weak form (LaTeX):\n"
            << assembler.as_latex(decorator) << std::endl;

    // Now we pass in concrete objects to get data from
    // and assemble into.
    assembler.assemble_matrix(system_matrix_wf,
                              constraints,
                              dof_handler,
                              qf_cell);

    // system_matrix_wf.print(std::cout);
    verify_assembly(system_matrix_std, system_matrix_wf);
  }

  // Tensor valued coefficient
  {
    using namespace WeakForms;

    deallog << "Weak form assembly (bilinear form, tensor valued coefficient)"
            << std::endl;
    system_matrix_wf = 0;

    // Symbolic types for test function, trial solution and a coefficient.
    const TestFunction<dim, spacedim>  test;
    const TrialSolution<dim, spacedim> trial;
    const TensorFunctor<0, spacedim>   coeff("C", "C");

    const auto test_lap   = test.laplacian();
    const auto trial_lap  = trial.laplacian();
    const auto coeff_func = coeff.template value<double, dim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      { return Tensor<0, spacedim, double>({1.0}); });

    // Still no concrete definitions
    MatrixBasedAssembler<dim, spacedim> assembler;
    assembler += bilinear_form(test_lap, coeff_func, trial_lap).dV();

    // Look at what we're going to compute
    const SymbolicDecorations decorator;
    deallog << "Weak form (ascii):\n"
            << assembler.as_ascii(decorator) << std::endl;
    deallog << "Weak form (LaTeX):\n"
            << assembler.as_latex(decorator) << std::endl;

    // Now we pass in concrete objects to get data from
    // and assemble into.
    assembler.assemble_matrix(system_matrix_wf,
                              constraints,
                              dof_handler,
                              qf_cell);

    // system_matrix_wf.print(std::cout);
    verify_assembly(system_matrix_std, system_matrix_wf);
  }

  deallog << "OK" << std::endl;
}


int
main(int argc, char *argv[])
{
  initlog();
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  run<2>();
  run<3>();

  deallog << "OK" << std::endl;
}
