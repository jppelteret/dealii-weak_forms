/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 - 2020 by the deal.II authors
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
 * Authors: Natasha Sharma, University of Texas at El Paso,
 *          Guido Kanschat, University of Heidelberg
 *          Timo Heister, Clemson University
 *          Wolfgang Bangerth, Colorado State University
 *          Zhuroan Wang, Colorado State University
 */



#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <fstream>
#include <iostream>


namespace Step47
{
  using namespace dealii;


  namespace ExactSolution
  {
    using numbers::PI;

    template <int dim>
    class Solution : public Function<dim>
    {
    public:
      static_assert(dim == 2, "Only dim==2 is implemented.");

      virtual double
      value(const Point<dim> &p,
            const unsigned int /*component*/ = 0) const override
      {
        return std::sin(PI * p[0]) * std::sin(PI * p[1]);
      }

      virtual Tensor<1, dim>
      gradient(const Point<dim> &p,
               const unsigned int /*component*/ = 0) const override
      {
        Tensor<1, dim> r;
        r[0] = PI * std::cos(PI * p[0]) * std::sin(PI * p[1]);
        r[1] = PI * std::cos(PI * p[1]) * std::sin(PI * p[0]);
        return r;
      }

      virtual void
      hessian_list(const std::vector<Point<dim>> &       points,
                   std::vector<SymmetricTensor<2, dim>> &hessians,
                   const unsigned int /*component*/ = 0) const override
      {
        for (unsigned i = 0; i < points.size(); ++i)
          {
            const double x = points[i][0];
            const double y = points[i][1];

            hessians[i][0][0] = -PI * PI * std::sin(PI * x) * std::sin(PI * y);
            hessians[i][0][1] = PI * PI * std::cos(PI * x) * std::cos(PI * y);
            hessians[i][1][1] = -PI * PI * std::sin(PI * x) * std::sin(PI * y);
          }
      }
    };


    template <int dim>
    class RightHandSide : public Function<dim>
    {
    public:
      static_assert(dim == 2, "Only dim==2 is implemented");

      virtual double
      value(const Point<dim> &p,
            const unsigned int /*component*/ = 0) const override

      {
        return 4 * std::pow(PI, 4.0) * std::sin(PI * p[0]) *
               std::sin(PI * p[1]);
      }
    };
  } // namespace ExactSolution



  template <int dim>
  class BiharmonicProblem
  {
  public:
    BiharmonicProblem(const unsigned int fe_degree);

    void
    run(const unsigned int n_local_refinement_levels,
        const unsigned int n_global_refinement_levels = 1);

  protected:
    void
    make_grid(const unsigned int n_global_refinement_levels);
    void
    setup_system();
    virtual void
    assemble_system() = 0;
    void
    solve();
    void
    compute_errors(const unsigned int iteration);
    void
    output_results(const unsigned int iteration) const;

    Triangulation<dim> triangulation;

    MappingQ<dim> mapping;

    FE_Q<dim>                 fe;
    DoFHandler<dim>           dof_handler;
    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> system_rhs;
  };



  template <int dim>
  BiharmonicProblem<dim>::BiharmonicProblem(const unsigned int fe_degree)
    : mapping(1)
    , fe(fe_degree)
    , dof_handler(triangulation)
  {}



  template <int dim>
  void
  BiharmonicProblem<dim>::make_grid(
    const unsigned int n_global_refinement_levels)
  {
    GridGenerator::hyper_cube(triangulation, 0., 1.);
    triangulation.refine_global(n_global_refinement_levels);

    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Total number of cells: " << triangulation.n_cells()
              << std::endl;
  }



  template <int dim>
  void
  BiharmonicProblem<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             ExactSolution::Solution<dim>(),
                                             constraints);
    constraints.close();


    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, constraints, true);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }



  template <int dim>
  struct ScratchData
  {
    ScratchData(const Mapping<dim> &      mapping,
                const FiniteElement<dim> &fe,
                const unsigned int        quadrature_degree,
                const UpdateFlags         update_flags,
                const UpdateFlags         interface_update_flags)
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
      , fe_interface_values(scratch_data.fe_values.get_mapping(),
                            scratch_data.fe_values.get_fe(),
                            scratch_data.fe_interface_values.get_quadrature(),
                            scratch_data.fe_interface_values.get_update_flags())
    {}

    FEValues<dim>          fe_values;
    FEInterfaceValues<dim> fe_interface_values;
  };



  struct CopyData
  {
    CopyData(const unsigned int dofs_per_cell)
      : cell_matrix(dofs_per_cell, dofs_per_cell)
      , cell_rhs(dofs_per_cell)
      , local_dof_indices(dofs_per_cell)
    {}


    CopyData(const CopyData &) = default;


    struct FaceData
    {
      FullMatrix<double>                   cell_matrix;
      std::vector<types::global_dof_index> joint_dof_indices;
    };

    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<FaceData>                face_data;
  };



  template <int dim>
  void
  BiharmonicProblem<dim>::solve()
  {
    std::cout << "   Solving system..." << std::endl;

    SparseDirectUMFPACK A_direct;
    A_direct.initialize(system_matrix);
    A_direct.vmult(solution, system_rhs);

    constraints.distribute(solution);
  }



  template <int dim>
  void
  BiharmonicProblem<dim>::compute_errors(const unsigned int iteration)
  {
    {
      Vector<float> norm_per_cell(triangulation.n_active_cells());
      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        solution,
                                        ExactSolution::Solution<dim>(),
                                        norm_per_cell,
                                        QGauss<dim>(fe.degree + 2),
                                        VectorTools::L2_norm);
      const double error_norm =
        VectorTools::compute_global_error(triangulation,
                                          norm_per_cell,
                                          VectorTools::L2_norm);
      deallog << "Iteration: " << iteration
              << "   Error in the L2 norm           :     " << error_norm
              << std::endl;
    }

    {
      Vector<float> norm_per_cell(triangulation.n_active_cells());
      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        solution,
                                        ExactSolution::Solution<dim>(),
                                        norm_per_cell,
                                        QGauss<dim>(fe.degree + 2),
                                        VectorTools::H1_seminorm);
      const double error_norm =
        VectorTools::compute_global_error(triangulation,
                                          norm_per_cell,
                                          VectorTools::H1_seminorm);
      deallog << "Iteration: " << iteration
              << "   Error in the H1 seminorm       : " << error_norm
              << std::endl;
    }

    {
      const QGauss<dim>            quadrature_formula(fe.degree + 2);
      ExactSolution::Solution<dim> exact_solution;
      Vector<double> error_per_cell(triangulation.n_active_cells());

      FEValues<dim> fe_values(mapping,
                              fe,
                              quadrature_formula,
                              update_values | update_hessians |
                                update_quadrature_points | update_JxW_values);

      FEValuesExtractors::Scalar scalar(0);
      const unsigned int         n_q_points = quadrature_formula.size();

      std::vector<SymmetricTensor<2, dim>> exact_hessians(n_q_points);
      std::vector<Tensor<2, dim>>          hessians(n_q_points);
      for (auto &cell : dof_handler.active_cell_iterators())
        {
          fe_values.reinit(cell);
          fe_values[scalar].get_function_hessians(solution, hessians);
          exact_solution.hessian_list(fe_values.get_quadrature_points(),
                                      exact_hessians);

          double local_error = 0;
          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              local_error +=
                ((exact_hessians[q_point] - hessians[q_point]).norm_square() *
                 fe_values.JxW(q_point));
            }
          error_per_cell[cell->active_cell_index()] = std::sqrt(local_error);
        }

      const double error_norm = error_per_cell.l2_norm();
      deallog << "Iteration: " << iteration
              << "   Error in the broken H2 seminorm: " << error_norm
              << std::endl;
    }
  }



  template <int dim>
  void
  BiharmonicProblem<dim>::output_results(const unsigned int iteration) const
  {
    constexpr bool output_vtk = false;

    if (output_vtk)
      {
        std::cout << "   Writing graphical output..." << std::endl;

        DataOut<dim> data_out;

        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution, "solution");
        data_out.build_patches();

        const std::string filename =
          ("output_" + Utilities::int_to_string(iteration, 6) + ".vtu");
        std::ofstream output_vtu(filename);
        data_out.write_vtu(output_vtu);
      }
  }



  template <int dim>
  void
  BiharmonicProblem<dim>::run(const unsigned int n_local_refinement_levels,
                              const unsigned int n_global_refinement_levels)
  {
    make_grid(n_global_refinement_levels);

    for (unsigned int cycle = 0; cycle < n_local_refinement_levels; ++cycle)
      {
        std::cout << "Cycle " << cycle << " of " << n_local_refinement_levels
                  << std::endl;

        triangulation.refine_global(1);
        setup_system();

        assemble_system();
        solve();

        output_results(cycle);

        compute_errors(cycle);
        std::cout << std::endl;
      }
  }
} // namespace Step47
