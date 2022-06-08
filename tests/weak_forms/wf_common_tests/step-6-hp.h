/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 */

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

// This header replicates a combination of step-6 and step-27,
// and leaves some aspects of its implementation out so that they may be
// modified.
// It is used as a baseline for the weak form tests.
// - hp variant


#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_series.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/refinement.h>

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
#include <deal.II/numerics/smoothness_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>


using namespace dealii;

template <int dim>
class RightHandSide : public Function<dim>
{
public:
  virtual double
  value(const Point<dim> &p, const unsigned int component) const override;
};


template <int dim>
double
RightHandSide<dim>::value(const Point<dim> &p,
                          const unsigned int /*component*/) const
{
  double product = 1;
  if (p.square() < 0.5 * 0.5)
    return 20;
  else
    return 1;
}


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

  const unsigned int min_degree;
  const unsigned int max_degree;

  hp::FECollection<dim>    fe_collection;
  hp::QCollection<dim>     qf_collection_cell;
  hp::QCollection<dim - 1> qf_collection_face;

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
  : min_degree(1)
  , max_degree(3)
  , dof_handler(triangulation)
{
  for (unsigned int degree = min_degree; degree <= max_degree; ++degree)
    {
      fe_collection.push_back(FE_Q<dim>(degree));
      qf_collection_cell.push_back(QGauss<dim>(degree + 1));
      qf_collection_face.push_back(QGauss<dim - 1>(degree + 1));
    }
}


template <int dim>
void
Step6_Base<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe_collection);

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
    dof_handler, qf_collection_face, {}, solution, estimated_error_per_cell);

  Vector<float>          smoothness_indicators(triangulation.n_active_cells());
  FESeries::Fourier<dim> fourier =
    SmoothnessEstimator::Fourier::default_fe_series(fe_collection);
  SmoothnessEstimator::Fourier::coefficient_decay(fourier,
                                                  dof_handler,
                                                  solution,
                                                  smoothness_indicators);

  GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                  estimated_error_per_cell,
                                                  0.3,
                                                  0.03);

  hp::Refinement::p_adaptivity_from_relative_threshold(dof_handler,
                                                       smoothness_indicators,
                                                       0.2,
                                                       0.2);

  hp::Refinement::choose_p_over_h(dof_handler);

  triangulation.prepare_coarsening_and_refinement();
  hp::Refinement::limit_p_level_difference(dof_handler);

  triangulation.execute_coarsening_and_refinement();
}


template <int dim>
void
Step6_Base<dim>::output_results(const unsigned int cycle) const
{
  constexpr bool output_vtu = false;
  if (output_vtu)
    {
      Vector<float> fe_degrees(triangulation.n_active_cells());
      for (const auto &cell : dof_handler.active_cell_iterators())
        fe_degrees(cell->active_cell_index()) =
          fe_collection[cell->active_fe_index()].degree;

      DataOut<dim> data_out;

      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;
      data_out.set_flags(flags);

      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution, "solution");
      data_out.add_data_vector(fe_degrees, "fe_degree");
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
