/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2008 - 2020 by the deal.II authors
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
 * Author: Liang Zhao and Timo Heister, Clemson University, 2016
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


// This header replicates step-57, leaves some aspects of its implementation
// out so that they may be modified.
// It is used as a baseline for the weak form tests.


#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

namespace Step57
{
  using namespace dealii;


  template <int dim>
  class StationaryNavierStokes
  {
  public:
    StationaryNavierStokes(const unsigned int degree);
    void
    run(const unsigned int refinement,
        const unsigned int n_global_refinements = 3);

  protected:
    void
    setup_dofs();

    void
    initialize_system();

    virtual void
    assemble(const bool initial_step, const bool assemble_matrix) = 0;

    void
    assemble_system(const bool initial_step);

    void
    assemble_rhs(const bool initial_step);

    void
    solve(const bool initial_step);

    void
    refine_mesh();

    void
    process_solution(unsigned int refinement);

    void
    output_results(const unsigned int refinement_cycle) const;

    void
    newton_iteration(const double       tolerance,
                     const unsigned int max_n_line_searches,
                     const unsigned int max_n_refinements,
                     const bool         is_initial_step,
                     const bool         output_result);

    void
    compute_initial_guess(double step_size);

    double                               viscosity;
    double                               gamma;
    const unsigned int                   degree;
    std::vector<types::global_dof_index> dofs_per_block;

    Triangulation<dim> triangulation;
    FESystem<dim>      fe;
    DoFHandler<dim>    dof_handler;

    AffineConstraints<double> zero_constraints;
    AffineConstraints<double> nonzero_constraints;

    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;
    SparseMatrix<double>      pressure_mass_matrix;

    BlockVector<double> present_solution;
    BlockVector<double> newton_update;
    BlockVector<double> system_rhs;
    BlockVector<double> evaluation_point;
  };


  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues()
      : Function<dim>(dim + 1)
    {}
    virtual double
    value(const Point<dim> &p, const unsigned int component) const override;
  };

  template <int dim>
  double
  BoundaryValues<dim>::value(const Point<dim> & p,
                             const unsigned int component) const
  {
    Assert(component < this->n_components,
           ExcIndexRange(component, 0, this->n_components));
    if (component == 0 && std::abs(p[dim - 1] - 1.0) < 1e-10)
      return 1.0;

    return 0;
  }

  template <class PreconditionerMp>
  class BlockSchurPreconditioner : public Subscriptor
  {
  public:
    BlockSchurPreconditioner(double                           gamma,
                             double                           viscosity,
                             const BlockSparseMatrix<double> &S,
                             const SparseMatrix<double> &     P,
                             const PreconditionerMp &         Mppreconditioner);

    void
    vmult(BlockVector<double> &dst, const BlockVector<double> &src) const;

  private:
    const double                     gamma;
    const double                     viscosity;
    const BlockSparseMatrix<double> &stokes_matrix;
    const SparseMatrix<double> &     pressure_mass_matrix;
    const PreconditionerMp &         mp_preconditioner;
    SparseDirectUMFPACK              A_inverse;
  };


  template <class PreconditionerMp>
  BlockSchurPreconditioner<PreconditionerMp>::BlockSchurPreconditioner(
    double                           gamma,
    double                           viscosity,
    const BlockSparseMatrix<double> &S,
    const SparseMatrix<double> &     P,
    const PreconditionerMp &         Mppreconditioner)
    : gamma(gamma)
    , viscosity(viscosity)
    , stokes_matrix(S)
    , pressure_mass_matrix(P)
    , mp_preconditioner(Mppreconditioner)
  {
    A_inverse.initialize(stokes_matrix.block(0, 0));
  }

  template <class PreconditionerMp>
  void
  BlockSchurPreconditioner<PreconditionerMp>::vmult(
    BlockVector<double> &      dst,
    const BlockVector<double> &src) const
  {
    Vector<double> utmp(src.block(0));

    {
      SolverControl            solver_control(1000,
                                   1e-6 * src.block(1).l2_norm(),
                                   false,
                                   false);
      SolverCG<Vector<double>> cg(solver_control);

      dst.block(1) = 0.0;
      cg.solve(pressure_mass_matrix,
               dst.block(1),
               src.block(1),
               mp_preconditioner);
      dst.block(1) *= -(viscosity + gamma);
    }

    {
      stokes_matrix.block(0, 1).vmult(utmp, dst.block(1));
      utmp *= -1.0;
      utmp += src.block(0);
    }

    A_inverse.vmult(dst.block(0), utmp);
  }

  template <int dim>
  StationaryNavierStokes<dim>::StationaryNavierStokes(const unsigned int degree)
    : viscosity(1.0 / 7500.0)
    , gamma(1.0)
    , degree(degree)
    , triangulation(Triangulation<dim>::maximum_smoothing)
    , fe(FE_Q<dim>(degree + 1), dim, FE_Q<dim>(degree), 1)
    , dof_handler(triangulation)
  {}

  template <int dim>
  void
  StationaryNavierStokes<dim>::setup_dofs()
  {
    system_matrix.clear();
    pressure_mass_matrix.clear();

    dof_handler.distribute_dofs(fe);

    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);

    dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    unsigned int dof_u = dofs_per_block[0];
    unsigned int dof_p = dofs_per_block[1];

    FEValuesExtractors::Vector velocities(0);
    {
      nonzero_constraints.clear();

      DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               BoundaryValues<dim>(),
                                               nonzero_constraints,
                                               fe.component_mask(velocities));
    }
    nonzero_constraints.close();

    {
      zero_constraints.clear();

      DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               Functions::ZeroFunction<dim>(
                                                 dim + 1),
                                               zero_constraints,
                                               fe.component_mask(velocities));
    }
    zero_constraints.close();

    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << " (" << dof_u << " + " << dof_p << ')' << std::endl;
  }

  template <int dim>
  void
  StationaryNavierStokes<dim>::initialize_system()
  {
    {
      BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
      DoFTools::make_sparsity_pattern(dof_handler, dsp, nonzero_constraints);
      sparsity_pattern.copy_from(dsp);
    }

    system_matrix.reinit(sparsity_pattern);

    present_solution.reinit(dofs_per_block);
    newton_update.reinit(dofs_per_block);
    system_rhs.reinit(dofs_per_block);
  }

  template <int dim>
  void
  StationaryNavierStokes<dim>::assemble_system(const bool initial_step)
  {
    assemble(initial_step, true);
  }

  template <int dim>
  void
  StationaryNavierStokes<dim>::assemble_rhs(const bool initial_step)
  {
    assemble(initial_step, false);
  }

  template <int dim>
  void
  StationaryNavierStokes<dim>::solve(const bool initial_step)
  {
    const AffineConstraints<double> &constraints_used =
      initial_step ? nonzero_constraints : zero_constraints;

    SolverControl solver_control(system_matrix.m(),
                                 1e-4 * system_rhs.l2_norm(),
                                 false,
                                 false);

    SolverFGMRES<BlockVector<double>> gmres(solver_control);
    SparseILU<double>                 pmass_preconditioner;
    pmass_preconditioner.initialize(pressure_mass_matrix,
                                    SparseILU<double>::AdditionalData());

    const BlockSchurPreconditioner<SparseILU<double>> preconditioner(
      gamma,
      viscosity,
      system_matrix,
      pressure_mass_matrix,
      pmass_preconditioner);

    gmres.solve(system_matrix, newton_update, system_rhs, preconditioner);
    std::cout << "FGMRES steps: " << solver_control.last_step() << std::endl;

    constraints_used.distribute(newton_update);
  }

  template <int dim>
  void
  StationaryNavierStokes<dim>::refine_mesh()
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    FEValuesExtractors::Vector velocity(0);
    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      QGauss<dim - 1>(degree + 1),
      std::map<dealii::types::boundary_id, const Function<dim> *>(),
      present_solution,
      estimated_error_per_cell,
      fe.component_mask(velocity));

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    estimated_error_per_cell,
                                                    0.3,
                                                    0.0);

    triangulation.prepare_coarsening_and_refinement();
    SolutionTransfer<dim, BlockVector<double>> solution_transfer(dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(present_solution);
    triangulation.execute_coarsening_and_refinement();

    setup_dofs();

    BlockVector<double> tmp(dofs_per_block);

    solution_transfer.interpolate(present_solution, tmp);
    nonzero_constraints.distribute(tmp);

    initialize_system();
    present_solution = tmp;
  }

  template <int dim>
  void
  StationaryNavierStokes<dim>::newton_iteration(
    const double       tolerance,
    const unsigned int max_n_line_searches,
    const unsigned int max_n_refinements,
    const bool         is_initial_step,
    const bool         output_result)
  {
    bool first_step = is_initial_step;

    for (unsigned int refinement_n = 0; refinement_n < max_n_refinements + 1;
         ++refinement_n)
      {
        unsigned int line_search_n = 0;
        double       last_res      = 1.0;
        double       current_res   = 1.0;
        std::cout << "grid refinements: " << refinement_n << std::endl
                  << "viscosity: " << viscosity << std::endl;

        while ((first_step || (current_res > tolerance)) &&
               line_search_n < max_n_line_searches)
          {
            if (first_step)
              {
                setup_dofs();
                initialize_system();
                evaluation_point = present_solution;
                assemble_system(first_step);
                solve(first_step);
                present_solution = newton_update;
                nonzero_constraints.distribute(present_solution);
                first_step       = false;
                evaluation_point = present_solution;
                assemble_rhs(first_step);
                current_res = system_rhs.l2_norm();
                std::cout << "The residual of initial guess is " << current_res
                          << std::endl;
                last_res = current_res;
              }
            else
              {
                evaluation_point = present_solution;
                assemble_system(first_step);
                solve(first_step);

                for (double alpha = 1.0; alpha > 1e-5; alpha *= 0.5)
                  {
                    evaluation_point = present_solution;
                    evaluation_point.add(alpha, newton_update);
                    nonzero_constraints.distribute(evaluation_point);
                    assemble_rhs(first_step);
                    current_res = system_rhs.l2_norm();
                    std::cout << "  alpha: " << std::setw(10) << alpha
                              << std::setw(0) << "  residual: " << current_res
                              << std::endl;
                    if (current_res < last_res)
                      break;
                  }
                {
                  present_solution = evaluation_point;
                  std::cout << "  number of line searches: " << line_search_n
                            << "  residual: " << current_res << std::endl;
                  last_res = current_res;
                }
                ++line_search_n;
              }

            if (output_result)
              {
                output_results(max_n_line_searches * refinement_n +
                               line_search_n);

                if (current_res <= tolerance)
                  process_solution(refinement_n);
              }
          }

        if (refinement_n < max_n_refinements)
          {
            refine_mesh();
          }
      }
  }

  template <int dim>
  void
  StationaryNavierStokes<dim>::compute_initial_guess(double step_size)
  {
    const double target_Re = 1.0 / viscosity;

    bool is_initial_step = true;

    for (double Re = 1000.0; Re < target_Re;
         Re        = std::min(Re + step_size, target_Re))
      {
        viscosity = 1.0 / Re;
        std::cout << "Searching for initial guess with Re = " << Re
                  << std::endl;
        newton_iteration(1e-12, 50, 0, is_initial_step, false);
        is_initial_step = false;
      }
  }

  template <int dim>
  void
  StationaryNavierStokes<dim>::output_results(
    const unsigned int output_index) const
  {
    const bool output_vtk = false;
    if (output_vtk)
      {
        std::vector<std::string> solution_names(dim, "velocity");
        solution_names.emplace_back("pressure");

        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          data_component_interpretation(
            dim, DataComponentInterpretation::component_is_part_of_vector);
        data_component_interpretation.push_back(
          DataComponentInterpretation::component_is_scalar);
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(present_solution,
                                 solution_names,
                                 DataOut<dim>::type_dof_data,
                                 data_component_interpretation);
        data_out.build_patches();

        std::ofstream output(std::to_string(1.0 / viscosity) + "-solution-" +
                             Utilities::int_to_string(output_index, 4) +
                             ".vtk");
        data_out.write_vtk(output);
      }
  }

  template <int dim>
  void
  StationaryNavierStokes<dim>::process_solution(unsigned int refinement)
  {
    deallog << std::to_string(1.0 / viscosity) + "-line-" +
                 std::to_string(refinement)
            << std::endl;

    Point<dim> p;
    p(0) = 0.5;
    p(1) = 0.5;

    for (unsigned int i = 0; i <= 100; ++i)
      {
        p(dim - 1) = i / 100.0;

        Vector<double> tmp_vector(dim + 1);
        VectorTools::point_value(dof_handler, present_solution, p, tmp_vector);
        deallog << p(dim - 1);

        for (int j = 0; j < dim; j++)
          deallog << " " << tmp_vector(j);
        deallog << std::endl;
      }
  }

  template <int dim>
  void
  StationaryNavierStokes<dim>::run(const unsigned int refinement,
                                   const unsigned int n_global_refinements)
  {
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(n_global_refinements);

    const double Re = 1.0 / viscosity;

    if (Re > 1000.0)
      {
        std::cout << "Searching for initial guess ..." << std::endl;
        const double step_size = 2000.0;
        compute_initial_guess(step_size);
        std::cout << "Found initial guess." << std::endl;
        std::cout << "Computing solution with target Re = " << Re << std::endl;
        viscosity = 1.0 / Re;
        newton_iteration(1e-12, 50, refinement, false, true);
      }
    else
      {
        newton_iteration(1e-12, 50, refinement, true, true);
      }
  }
} // namespace Step57
