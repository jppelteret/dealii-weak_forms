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

// This header replicates step-6, leaves some aspects of its implementation
// out so that they may be modified.
// It is used as a baseline for the weak form tests.


#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>


using namespace dealii;


template <int dim>
class Step6_Base
{
public:
  Step6_Base();

  void
  run();

protected:
  void
  setup_system();
  virtual void
  assemble_system() = 0;
  void
  solve();
  void
  refine_grid();
  void
  output_results(const unsigned int cycle) const;

  const FE_Q<dim>       fe;
  const QGauss<dim>     qf_cell;
  const QGauss<dim - 1> qf_face;

  Triangulation<dim> triangulation;
  DoFHandler<dim>    dof_handler;

  AffineConstraints<double> constraints;

  SparseMatrix<double> system_matrix;
  SparsityPattern      sparsity_pattern;

  Vector<double> solution;
  Vector<double> system_rhs;
};


template <int dim>
Step6_Base<dim>::Step6_Base()
  : fe(2)
  , qf_cell(fe.degree + 1)
  , qf_face(fe.degree + 1)
  , dof_handler(triangulation)
{}


template <int dim>
void
Step6_Base<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);


  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints);

  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);

  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
}


template <int dim>
void
Step6_Base<dim>::solve()
{
  SolverControl solver_control(system_matrix.m(), 1e-12, false, false);
  SolverCG<Vector<double>> solver(solver_control);

  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  constraints.distribute(solution);
}


template <int dim>
void
Step6_Base<dim>::refine_grid()
{
  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

  KellyErrorEstimator<dim>::estimate(
    dof_handler, qf_face, {}, solution, estimated_error_per_cell);

  GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                  estimated_error_per_cell,
                                                  0.3,
                                                  0.03);

  triangulation.execute_coarsening_and_refinement();
}


template <int dim>
void
Step6_Base<dim>::output_results(const unsigned int cycle) const
{
  constexpr bool output_vtu = false;
  if (output_vtu)
    {
      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution, "solution");
      data_out.build_patches();

      std::ofstream output("solution-" + std::to_string(cycle) + ".vtu");
      data_out.write_vtu(output);
    }

  {
    const Point<dim>                      soln_pt;
    const Functions::FEFieldFunction<dim> fe_field_function(dof_handler,
                                                            solution);
    deallog << "Cycle " << cycle << ": " << fe_field_function.value(soln_pt)
            << std::endl;
  }
}


template <int dim>
void
Step6_Base<dim>::run()
{
  for (unsigned int cycle = 0; cycle < 8; ++cycle)
    {
      std::cout << "Cycle " << cycle << ':' << std::endl;

      if (cycle == 0)
        {
          GridGenerator::hyper_ball(triangulation);
          triangulation.refine_global(1);
        }
      else
        refine_grid();


      std::cout << "   Number of active cells:       "
                << triangulation.n_active_cells() << std::endl;

      setup_system();

      std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                << std::endl;

      assemble_system();
      solve();
      output_results(cycle);
    }
}
