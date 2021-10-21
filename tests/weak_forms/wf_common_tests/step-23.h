// ---------------------------------------------------------------------
//
// Copyright (C) 2020 - 2021 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------


// This header replicates step-23, leaves some aspects of its implementation
// out so that they may be modified.
// It is used as a baseline for the weak form tests.
//
// This header is based off of:
// tests/simplex/step-23.cc


#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

const unsigned int degree = 1;

namespace Step23
{
  using namespace dealii;

  template <int dim>
  class Step23_Base
  {
  public:
    Step23_Base();
    void
    run();

  protected:
    void
    setup_system();
    void
    assemble_forcing_terms(Vector<double> &forcing_terms);
    virtual void
    assemble_u() = 0;
    void
    solve_u();
    virtual void
    assemble_v() = 0;
    void
    solve_v();
    void
    output_results() const;

    Triangulation<dim> triangulation;
    MappingQ<dim, dim> mapping;
    FE_Q<dim>          fe;
    QGauss<dim>        quadrature;
    DoFHandler<dim>    dof_handler;

    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;
    SparseMatrix<double> matrix_u;
    SparseMatrix<double> matrix_v;

    Vector<double> solution_u, solution_v;
    Vector<double> old_solution_u, old_solution_v;
    Vector<double> system_rhs;

    double       time_step;
    double       time;
    unsigned int timestep_number;
    const double theta;
  };

  template <int dim>
  class InitialValuesU : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));
      return 0;
    }
  };



  template <int dim>
  class InitialValuesV : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));
      return 0;
    }
  };



  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));
      return 0;
    }
  };



  template <int dim>
  class BoundaryValuesU : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      if ((this->get_time() <= 0.5) && (p[0] < 0) && (p[1] < 1. / 3) &&
          (p[1] > -1. / 3))
        return std::sin(this->get_time() * 4 * numbers::PI);
      else
        return 0;
    }
  };



  template <int dim>
  class BoundaryValuesV : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      if ((this->get_time() <= 0.5) && (p[0] < 0) && (p[1] < 1. / 3) &&
          (p[1] > -1. / 3))
        return (std::cos(this->get_time() * 4 * numbers::PI) * 4 * numbers::PI);
      else
        return 0;
    }
  };

  template <int dim>
  Step23_Base<dim>::Step23_Base()
    : mapping(1)
    , fe(1)
    , quadrature(fe.degree + 1)
    , dof_handler(triangulation)
    , time_step(1. / 64)
    , time(time_step)
    , timestep_number(1)
    , theta(0.5)
  {}

  template <int dim>
  void
  Step23_Base<dim>::setup_system()
  {
    const unsigned int n_subdivisions = (1 << 3);
    GridGenerator::subdivided_hyper_cube(triangulation, n_subdivisions, -1, 1);

    deallog << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;

    dof_handler.distribute_dofs(fe);

    deallog << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl
            << std::endl;

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    matrix_u.reinit(sparsity_pattern);
    matrix_v.reinit(sparsity_pattern);

    MatrixCreator::create_mass_matrix(mapping,
                                      dof_handler,
                                      quadrature,
                                      mass_matrix);
    MatrixCreator::create_laplace_matrix(mapping,
                                         dof_handler,
                                         quadrature,
                                         laplace_matrix);
    solution_u.reinit(dof_handler.n_dofs());
    solution_v.reinit(dof_handler.n_dofs());
    old_solution_u.reinit(dof_handler.n_dofs());
    old_solution_v.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    constraints.close();
  }


  template <int dim>
  void
  Step23_Base<dim>::assemble_forcing_terms(Vector<double> &forcing_terms)
  {
    Vector<double> tmp(solution_u.size());

    RightHandSide<dim> rhs_function;
    rhs_function.set_time(this->time);
    VectorTools::create_right_hand_side(
      this->mapping, this->dof_handler, this->quadrature, rhs_function, tmp);
    forcing_terms = tmp;
    forcing_terms *= this->theta * this->time_step;

    rhs_function.set_time(time - time_step);
    VectorTools::create_right_hand_side(
      this->mapping, this->dof_handler, this->quadrature, rhs_function, tmp);

    forcing_terms.add((1 - this->theta) * this->time_step, tmp);
  }



  template <int dim>
  void
  Step23_Base<dim>::solve_u()
  {
    SolverControl            solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    cg.solve(matrix_u, solution_u, system_rhs, PreconditionIdentity());

    deallog << "   u-equation: " << solver_control.last_step()
            << " CG iterations." << std::endl;
  }



  template <int dim>
  void
  Step23_Base<dim>::solve_v()
  {
    SolverControl            solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    cg.solve(matrix_v, solution_v, system_rhs, PreconditionIdentity());

    deallog << "   v-equation: " << solver_control.last_step()
            << " CG iterations." << std::endl;
  }


  template <int dim>
  void
  Step23_Base<dim>::output_results() const
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution_u, "U");
    data_out.add_data_vector(solution_v, "V");

    data_out.build_patches(mapping);

    const std::string filename =
      "solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtu";
    DataOutBase::VtkFlags vtk_flags;
    vtk_flags.compression_level =
      DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
    data_out.set_flags(vtk_flags);
    std::ofstream output(filename);
    data_out.write_vtu(output);
  }


  template <int dim>
  void
  Step23_Base<dim>::run()
  {
    setup_system();

    VectorTools::project(mapping,
                         dof_handler,
                         constraints,
                         quadrature,
                         InitialValuesU<dim>(),
                         old_solution_u);
    VectorTools::project(mapping,
                         dof_handler,
                         constraints,
                         quadrature,
                         InitialValuesV<dim>(),
                         old_solution_v);

    for (; time <= 5; time += time_step, ++timestep_number)
      {
        deallog << "Time step " << timestep_number << " at t=" << time
                << std::endl;

        assemble_u();
        {
          BoundaryValuesU<dim> boundary_values_u_function;
          boundary_values_u_function.set_time(time);

          std::map<types::global_dof_index, double> boundary_values;
          VectorTools::interpolate_boundary_values(mapping,
                                                   dof_handler,
                                                   0,
                                                   boundary_values_u_function,
                                                   boundary_values);
          MatrixTools::apply_boundary_values(boundary_values,
                                             matrix_u,
                                             solution_u,
                                             system_rhs);
        }
        solve_u();

        assemble_v();
        {
          BoundaryValuesV<dim> boundary_values_v_function;
          boundary_values_v_function.set_time(time);

          std::map<types::global_dof_index, double> boundary_values;
          VectorTools::interpolate_boundary_values(mapping,
                                                   dof_handler,
                                                   0,
                                                   boundary_values_v_function,
                                                   boundary_values);
          MatrixTools::apply_boundary_values(boundary_values,
                                             matrix_v,
                                             solution_v,
                                             system_rhs);
        }
        solve_v();

        constexpr bool plot_output = false;
        if (plot_output)
          {
            output_results();
          }

        deallog << "   Total energy: "
                << (mass_matrix.matrix_norm_square(solution_v) +
                    laplace_matrix.matrix_norm_square(solution_u)) /
                     2
                << std::endl;

        old_solution_u = solution_u;
        old_solution_v = solution_v;
      }
  }
} // namespace Step23
