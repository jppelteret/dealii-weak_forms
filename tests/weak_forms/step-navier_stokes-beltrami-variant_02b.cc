/* $Id: NavierStokes-Beltrami.cc 2008-04-15 10:54:52CET martinkr $ */
/* Author: Martin Kronbichler, Uppsala University, 2008 */
/*    $Id: NavierStokes-Beltrami.cc 2008-04-15 10:54:52CET martinkr $ */
/*    Version: $Name$                                             */
/*                                                                */
/*    Copyright (C) 2008 by the author                            */
/*                                                                */
/*    This file is subject to QPL and may not be  distributed     */
/*    without copyright and license information. Please refer     */
/*    to the file deal.II/doc/license.html for the  text  and     */
/*    further information on this license.                        */

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

// This program implements an algorithm for the incompressible Navier-Stokes
// equations solving the whole system at once, i.e., without any projection
// equation.
// This test replicates step-navier_stokes-beltrami exactly.
//
// This variant has maximum stablisation enabled.

#include "../weak_forms_tests.h"
#include "wf_common_tests/step-navier_stokes-beltrami.h"

namespace StepNavierStokesBeltrami
{
  template <int dim>
  class NavierStokesProblem : public NavierStokesProblemBase<dim>
  {
  public:
    NavierStokesProblem()
      : NavierStokesProblemBase<dim>(5 /*stabilization*/)
    {}

  protected:
    void
    assemble_system() override;
  };

  template <int dim>
  void
  NavierStokesProblem<dim>::assemble_system()
  {
    this->system_matrix = 0;
    this->system_rhs    = 0;

    // The function that actually
    // assembles the linear system is
    // similar to the one in step-22
    // of the deal.II tutorial.
    // We allocate some variables
    // and build the structure
    // for the cell stiffness matrix
    // <code>local_matrix</code> and the
    // local right hand side vector
    // <code>local_rhs</code>.
    // As usual, <code>local_dof_indices</code>
    // contains the information on
    // the position of the cell
    // degrees of freedom in the
    // global system.
    const QGauss<dim> quadrature_formula(this->degree + 2);

    FEValues<dim> fe_values(this->fe,
                            quadrature_formula,
                            update_values | update_gradients | update_hessians |
                              update_inverse_jacobians |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = this->fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<unsigned int> local_dof_indices(dofs_per_cell);

    // We declare the objects that contain
    // the information on the right hand
    // side of the Navier-Stokes equations and
    // the boudary values on both velocity and pressure.
    // We need both a representation for the continuous
    // functions as well as their values
    // at the quadrature points in the cells and
    // on the faces, respectively.
    const RightHandSide<dim>    right_hand_side(dim, this->time);
    std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(dim));

    // Vectors to hold the evaluations of
    // the FE basis functions and derived
    // quantities in the Navier-Stokes
    // equations assembly. The names should
    // mostly be self-explaining. The terms
    // usually include also weights from
    // the quadrature formula, so one needs
    // to be careful when applying the
    // terms in other orders as the ones
    // used in the code.
    //
    // Next follow similar arrays for
    // the evalutions of the right hand
    // side at the quadrature points,
    // i.e. forcing function and information
    // from the old time step.
    std::vector<std::vector<double>> phi_u(dofs_per_cell,
                                           std::vector<double>(n_q_points));
    std::vector<std::vector<double>> phi_u_weight(
      dofs_per_cell, std::vector<double>(n_q_points));
    std::vector<std::vector<Tensor<1, dim>>> grad_phi_u(
      dofs_per_cell, std::vector<Tensor<1, dim>>(n_q_points));
    std::vector<std::vector<double>> gradT_phi_u(
      dofs_per_cell, std::vector<double>(n_q_points));
    std::vector<std::vector<double>> div_phi_u_p(
      dofs_per_cell, std::vector<double>(n_q_points));
    std::vector<std::vector<double>> residual_phi(
      dofs_per_cell, std::vector<double>(n_q_points));
    std::vector<std::vector<Tensor<1, dim>>> stab_grad_phi(
      dofs_per_cell, std::vector<Tensor<1, dim>>(n_q_points));

    std::vector<Tensor<1, dim>> func_rhs(n_q_points);
    std::vector<Tensor<1, dim>> stab_rhs(n_q_points);

    // With all this in place, we can
    // go on with the loop over all
    // cells and add the local contributions.
    //
    // The first thing to do is to
    // evaluate the FE basis functions
    // at the quadrature
    // points of the cell, as well as
    // derivatives and the other
    // quantities specified above.
    // Moreover, we need to reset
    // the local matrices and
    // right hand side before
    // filling them with new information
    // from the current cell.
    for (const auto &cell : this->dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        local_matrix = 0;
        local_rhs    = 0;
        cell->get_dof_indices(local_dof_indices);
        const double h = cell->diameter();

        // Evalulate right hand side function,
        // density and viscosity at the
        // integration points. Then, we use
        // the finite element basis
        // functions to get the interpolated
        // velocity value at the quadrature
        // points. We do this for both
        // the current velocity and
        // the (weighted) old velocity.
        right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                          rhs_values);

        // For building the total matrix,
        // start by translating the general
        // basis functions into the relevant
        // information for the Navier-Stokes
        // system.
        get_fe(fe_values,
               this->solution,
               this->solution_old_scaled,
               local_dof_indices,
               this->nu,
               rhs_values,
               this->time_step_weight,
               this->stabilization,
               h,
               phi_u,
               phi_u_weight,
               grad_phi_u,
               gradT_phi_u,
               div_phi_u_p,
               residual_phi,
               stab_grad_phi,
               func_rhs,
               stab_rhs);

        // Loop over the cell dofs in
        // order to fill the local matrix
        // with information. Note that we
        // loop first over the dofs and
        // only at the innermost position
        // over the quadrature points. This
        // accelerates the code for this
        // example, since we need less
        // <code>if</code> statements in
        // order to determine which terms
        // we need to calculate for the
        // current result.
        //
        // We also include different assembly
        // options, depending on whether we
        // have stabilization turned on
        // or off. This gives us a code that
        // is more difficult to read, but it
        // avoids a performance penalty we'd
        // face otherwise when no stabilization
        // would be used.
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const unsigned int component_i =
              this->fe.system_to_component_index(i).first;
            if (component_i < dim && this->stabilization < 2)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    const unsigned int component_j =
                      this->fe.system_to_component_index(j).first;

                    if (component_j < dim && component_i == component_j)
                      for (unsigned int q = 0; q < n_q_points; ++q)
                        local_matrix(i, j) +=
                          phi_u[i][q] * phi_u_weight[j][q] +
                          grad_phi_u[i][q] * grad_phi_u[j][q] +
                          gradT_phi_u[i][q] * gradT_phi_u[j][q];

                    else if (component_j < dim)
                      for (unsigned int q = 0; q < n_q_points; ++q)
                        local_matrix(i, j) +=
                          gradT_phi_u[i][q] * gradT_phi_u[j][q];

                    else if (component_j == dim)
                      for (unsigned int q = 0; q < n_q_points; ++q)
                        local_matrix(i, j) -=
                          div_phi_u_p[i][q] * div_phi_u_p[j][q];
                  }

                for (unsigned int q = 0; q < n_q_points; ++q)
                  local_rhs(i) += phi_u[i][q] * func_rhs[q][component_i];
              } /* end case for velocity dofs w/o stabilization */
            else if (component_i < dim)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    const unsigned int component_j =
                      this->fe.system_to_component_index(j).first;

                    if (component_j < dim && component_j == component_i)
                      for (unsigned int q = 0; q < n_q_points; ++q)
                        local_matrix(i, j) +=
                          phi_u[i][q] * phi_u_weight[j][q] +
                          grad_phi_u[i][q] * grad_phi_u[j][q] +
                          gradT_phi_u[i][q] * gradT_phi_u[j][q] +
                          div_phi_u_p[i][q] * div_phi_u_p[j][q] +
                          stab_grad_phi[i][q][component_j] * residual_phi[j][q];

                    else if (component_j < dim)
                      for (unsigned int q = 0; q < n_q_points; ++q)
                        local_matrix(i, j) +=
                          gradT_phi_u[i][q] * gradT_phi_u[j][q] +
                          div_phi_u_p[i][q] * div_phi_u_p[j][q] +
                          stab_grad_phi[i][q][component_j] * residual_phi[j][q];

                    else if (component_j == dim)
                      for (unsigned int q = 0; q < n_q_points; ++q)
                        local_matrix(i, j) +=
                          -div_phi_u_p[i][q] * div_phi_u_p[j][q] +
                          stab_grad_phi[i][q] * stab_grad_phi[j][q];
                  }

                for (unsigned int q = 0; q < n_q_points; ++q)
                  local_rhs(i) += phi_u[i][q] * func_rhs[q][component_i] +
                                  stab_grad_phi[i][q] * stab_rhs[q];
              } /* end case for velocity dofs w/ stabilization */
            else if (component_i == dim && this->stabilization % 2 == 0)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    const unsigned int component_j =
                      this->fe.system_to_component_index(j).first;
                    if (component_j < dim)
                      for (unsigned int q = 0; q < n_q_points; ++q)
                        local_matrix(i, j) +=
                          div_phi_u_p[i][q] * div_phi_u_p[j][q];
                  }
              } /* end case for pressure dofs w/o stabilization */
            else if (component_i == dim)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    const unsigned int component_j =
                      this->fe.system_to_component_index(j).first;
                    if (component_j < dim)
                      for (unsigned int q = 0; q < n_q_points; ++q)
                        local_matrix(i, j) +=
                          div_phi_u_p[i][q] * div_phi_u_p[j][q] +
                          stab_grad_phi[i][q][component_j] * residual_phi[j][q];
                    else if (component_j == dim)
                      for (unsigned int q = 0; q < n_q_points; ++q)
                        local_matrix(i, j) +=
                          stab_grad_phi[i][q] * stab_grad_phi[j][q];
                  }

                for (unsigned int q = 0; q < n_q_points; ++q)
                  local_rhs(i) += stab_grad_phi[i][q] * stab_rhs[q];
              } /* end case for pressure dofs w/ stabilization */
          }     /* end loop over i in dofs_per_cell */

        // The final stage in the loop over all cells is to
        // insert the local contributions into the global
        // stiffness matrix <code>system_matrix</code> and
        // the global right hand side
        // <code>system_rhs</code>. Since there may be
        // several processes running in parallel, we have to
        // aquire permission to write into the global matrix
        // by the <code>assembler_lock</code> variable (in
        // order to not access the global matrix by two
        // processes simulaneously and possibly lose data).
        this->hanging_node_and_pressure_constraints.distribute_local_to_global(
          local_matrix, local_dof_indices, this->system_matrix);
        this->hanging_node_and_pressure_constraints.distribute_local_to_global(
          local_rhs, local_dof_indices, this->system_rhs);
      }

    this->hanging_node_and_pressure_constraints.condense(this->system_matrix);
    this->hanging_node_and_pressure_constraints.condense(this->system_rhs);
  }

} // namespace StepNavierStokesBeltrami


int
main(int argc, char **argv)
{
  initlog();
  deallog << std::setprecision(9);

  deallog.depth_file(1);

  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  using namespace dealii;
  try
    {
      const int                                          dim = 2;
      StepNavierStokesBeltrami::NavierStokesProblem<dim> navier_stokes_problem;
      navier_stokes_problem.run();
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
