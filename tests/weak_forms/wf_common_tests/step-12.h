/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2009 - 2020 by the deal.II authors
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
 * Author: Guido Kanschat, Texas A&M University, 2009
 *         Timo Heister, Clemson University, 2019
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


// This header replicates step-12, leaves some aspects of its implementation
// out so that they may be modified.
// It is used as a baseline for the weak form tests.


#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>


namespace Step12
{
  using namespace dealii;

  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues() = default;
    virtual void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<double> &          values,
               const unsigned int             component = 0) const override;
  };

  template <int dim>
  void
  BoundaryValues<dim>::value_list(const std::vector<Point<dim>> &points,
                                  std::vector<double> &          values,
                                  const unsigned int component) const
  {
    (void)component;
    AssertIndexRange(component, 1);
    Assert(values.size() == points.size(),
           ExcDimensionMismatch(values.size(), points.size()));

    for (unsigned int i = 0; i < values.size(); ++i)
      {
        if (points[i](0) < 0.5)
          values[i] = 1.;
        else
          values[i] = 0.;
      }
  }


  template <int dim>
  Tensor<1, dim>
  beta(const Point<dim> &p)
  {
    Assert(dim >= 2, ExcNotImplemented());

    Point<dim> wind_field;
    wind_field(0) = -p(1);
    wind_field(1) = p(0);

    if (wind_field.norm() > 1e-10)
      wind_field /= wind_field.norm();

    return wind_field;
  }

  template <int dim>
  class Beta : public TensorFunction<1, dim>
  {
  public:
    Beta() = default;
    virtual Tensor<1, dim>
    value(const Point<dim> &p) const override
    {
      return beta(p);
    }
  };


  template <int dim>
  struct ScratchData
  {
    ScratchData(const Mapping<dim> &      mapping,
                const FiniteElement<dim> &fe,
                const unsigned int        quadrature_degree,
                const UpdateFlags         update_flags = update_values |
                                                 update_gradients |
                                                 update_quadrature_points |
                                                 update_JxW_values,
                const UpdateFlags interface_update_flags =
                  update_values | update_gradients | update_quadrature_points |
                  update_JxW_values | update_normal_vectors)
      : fe_values(mapping, fe, QGauss<dim>(quadrature_degree), update_flags)
      , fe_interface_values(mapping,
                            fe,
                            QGauss<dim - 1>(quadrature_degree),
                            interface_update_flags)
    {}


    ScratchData(const ScratchData<dim> &scratch_data)
      : fe_values(scratch_data.fe_values.get_mapping(),
                  scratch_data.fe_values.get_fe(),
                  scratch_data.fe_values.get_quadrature(),
                  scratch_data.fe_values.get_update_flags())
      , fe_interface_values(scratch_data.fe_interface_values.get_mapping(),
                            scratch_data.fe_interface_values.get_fe(),
                            scratch_data.fe_interface_values.get_quadrature(),
                            scratch_data.fe_interface_values.get_update_flags())
    {}

    FEValues<dim>          fe_values;
    FEInterfaceValues<dim> fe_interface_values;
  };



  struct CopyDataFace
  {
    FullMatrix<double>                   cell_matrix;
    std::vector<types::global_dof_index> joint_dof_indices;
  };



  struct CopyData
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<CopyDataFace>            face_data;

    template <class Iterator>
    void
    reinit(const Iterator &cell, unsigned int dofs_per_cell)
    {
      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_rhs.reinit(dofs_per_cell);

      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
    }
  };


  template <int dim>
  class AdvectionProblem
  {
  public:
    AdvectionProblem();
    void
    run(const unsigned int n_local_refinement_levels,
        const unsigned int n_global_refinement_levels = 3);

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

    Triangulation<dim>   triangulation;
    const MappingQ1<dim> mapping;

    FE_DGQ<dim>     fe;
    DoFHandler<dim> dof_handler;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> right_hand_side;
  };


  template <int dim>
  AdvectionProblem<dim>::AdvectionProblem()
    : mapping()
    , fe(1)
    , dof_handler(triangulation)
  {}


  template <int dim>
  void
  AdvectionProblem<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);


    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    right_hand_side.reinit(dof_handler.n_dofs());
  }


  template <int dim>
  void
  AdvectionProblem<dim>::solve()
  {
    SolverControl                    solver_control(1000, 1e-12, false, false);
    SolverRichardson<Vector<double>> solver(solver_control);

    PreconditionBlockSSOR<SparseMatrix<double>> preconditioner;

    preconditioner.initialize(system_matrix, fe.n_dofs_per_cell());

    solver.solve(system_matrix, solution, right_hand_side, preconditioner);

    std::cout << "  Solver converged in " << solver_control.last_step()
              << " iterations." << std::endl;
  }


  template <int dim>
  void
  AdvectionProblem<dim>::refine_grid()
  {
    Vector<float> gradient_indicator(triangulation.n_active_cells());

    DerivativeApproximation::approximate_gradient(mapping,
                                                  dof_handler,
                                                  solution,
                                                  gradient_indicator);

    unsigned int cell_no = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      gradient_indicator(cell_no++) *=
        std::pow(cell->diameter(), 1 + 1.0 * dim / 2);

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    gradient_indicator,
                                                    0.3,
                                                    0.1);

    triangulation.execute_coarsening_and_refinement();
  }


  template <int dim>
  void
  AdvectionProblem<dim>::output_results(const unsigned int cycle) const
  {
    constexpr bool output_vtk = false;

    if (output_vtk)
      {
        const std::string filename =
          "solution-" + std::to_string(cycle) + ".vtk";
        std::cout << "  Writing solution to <" << filename << ">" << std::endl;
        std::ofstream output(filename);

        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution, "u", DataOut<dim>::type_dof_data);

        data_out.build_patches();

        data_out.write_vtk(output);
      }

    {
      Vector<float> values(triangulation.n_active_cells());
      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        solution,
                                        Functions::ZeroFunction<dim>(),
                                        values,
                                        QGauss<dim>(fe.degree + 1),
                                        VectorTools::Linfty_norm);
      const double l_infty =
        VectorTools::compute_global_error(triangulation,
                                          values,
                                          VectorTools::Linfty_norm);
      deallog << "Cycle: " << cycle << "  L-infinity norm: " << l_infty
              << std::endl;
    }
  }


  template <int dim>
  void
  AdvectionProblem<dim>::run(const unsigned int n_local_refinement_levels,
                             const unsigned int n_global_refinement_levels)
  {
    for (unsigned int cycle = 0; cycle < n_local_refinement_levels; ++cycle)
      {
        std::cout << "Cycle " << cycle << std::endl;

        if (cycle == 0)
          {
            GridGenerator::hyper_cube(triangulation);
            triangulation.refine_global(n_global_refinement_levels);
          }
        else
          refine_grid();

        std::cout << "  Number of active cells:       "
                  << triangulation.n_active_cells() << std::endl;

        setup_system();

        std::cout << "  Number of degrees of freedom: " << dof_handler.n_dofs()
                  << std::endl;

        assemble_system();
        solve();

        output_results(cycle);
      }
  }
} // namespace Step12
