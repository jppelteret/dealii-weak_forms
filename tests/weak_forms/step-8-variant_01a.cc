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

// This test replicates step-8.
// It is used as a baseline for the weak form tests.


#include "../weak_forms_tests.h"
#include "wf_common_tests/step-8.h"


using namespace dealii;



template <int dim>
class Step8 : public Step8_Base<dim>
{
public:
  Step8();

protected:
  void
  assemble_system() override;
};


template <int dim>
Step8<dim>::Step8()
  : Step8_Base<dim>()
{}


template <int dim>
void
Step8<dim>::assemble_system()
{
  QGauss<dim> quadrature_formula(this->fe.degree + 1);

  FEValues<dim> fe_values(this->fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = this->fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> lambda_values(n_q_points);
  std::vector<double> mu_values(n_q_points);

  const Functions::ConstantFunction<dim> lambda(1.), mu(1.);

  std::vector<Tensor<1, dim>> rhs_values(n_q_points);

  for (const auto &cell : this->dof_handler.active_cell_iterators())
    {
      cell_matrix = 0;
      cell_rhs    = 0;

      fe_values.reinit(cell);

      lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
      mu.value_list(fe_values.get_quadrature_points(), mu_values);
      right_hand_side(fe_values.get_quadrature_points(), rhs_values);

      for (const unsigned int i : fe_values.dof_indices())
        {
          const unsigned int component_i =
            this->fe.system_to_component_index(i).first;

          for (const unsigned int j : fe_values.dof_indices())
            {
              const unsigned int component_j =
                this->fe.system_to_component_index(j).first;

              for (const unsigned int q_point :
                   fe_values.quadrature_point_indices())
                {
                  cell_matrix(i, j) +=
                    (                                                  //
                      (fe_values.shape_grad(i, q_point)[component_i] * //
                       fe_values.shape_grad(j, q_point)[component_j] * //
                       lambda_values[q_point])                         //
                      +                                                //
                      (fe_values.shape_grad(i, q_point)[component_j] * //
                       fe_values.shape_grad(j, q_point)[component_i] * //
                       mu_values[q_point])                             //
                      +                                                //
                      ((component_i == component_j) ?                  //
                         (fe_values.shape_grad(i, q_point) *           //
                          fe_values.shape_grad(j, q_point) *           //
                          mu_values[q_point]) :                        //
                         0)                                            //
                      ) *                                              //
                    fe_values.JxW(q_point);                            //
                }
            }
        }

      for (const unsigned int i : fe_values.dof_indices())
        {
          const unsigned int component_i =
            this->fe.system_to_component_index(i).first;

          for (const unsigned int q_point :
               fe_values.quadrature_point_indices())
            cell_rhs(i) += fe_values.shape_value(i, q_point) *
                           rhs_values[q_point][component_i] *
                           fe_values.JxW(q_point);
        }

      cell->get_dof_indices(local_dof_indices);
      this->constraints.distribute_local_to_global(cell_matrix,
                                                   cell_rhs,
                                                   local_dof_indices,
                                                   this->system_matrix,
                                                   this->system_rhs);
    }
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
      Step8<2> elastic_problem_2d;
      elastic_problem_2d.run();
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
