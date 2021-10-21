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


// This test replicates step-23.
// It is used as a baseline for the weak form tests.

#include "../weak_forms_tests.h"
#include "wf_common_tests/step-23.h"

namespace Step23
{
  using namespace dealii;

  template <int dim>
  class Step23 : public Step23_Base<dim>
  {
  public:
    Step23()
      : Step23_Base<dim>()
    {}

  protected:
    void
    assemble_u() override;
    void
    assemble_v() override;
  };

  template <int dim>
  void
  Step23<dim>::assemble_u()
  {
    this->mass_matrix.vmult(this->system_rhs, this->old_solution_u);

    Vector<double> tmp(this->solution_u.size());
    this->mass_matrix.vmult(tmp, this->old_solution_v);
    this->system_rhs.add(this->time_step, tmp);

    this->laplace_matrix.vmult(tmp, this->old_solution_u);
    this->system_rhs.add(-this->theta * (1 - this->theta) * this->time_step *
                           this->time_step,
                         tmp);

    Vector<double> forcing_terms(this->solution_u.size());
    this->assemble_forcing_terms(forcing_terms);

    this->system_rhs.add(this->theta * this->time_step, forcing_terms);

    this->matrix_u.copy_from(this->mass_matrix);
    this->matrix_u.add(this->theta * this->theta * this->time_step *
                         this->time_step,
                       this->laplace_matrix);
  }

  template <int dim>
  void
  Step23<dim>::assemble_v()
  {
    this->laplace_matrix.vmult(this->system_rhs, this->solution_u);
    this->system_rhs *= -this->theta * this->time_step;

    Vector<double> tmp(this->solution_u.size());
    this->mass_matrix.vmult(tmp, this->old_solution_v);
    this->system_rhs += tmp;

    this->laplace_matrix.vmult(tmp, this->old_solution_u);
    this->system_rhs.add(-this->time_step * (1 - this->theta), tmp);

    Vector<double> forcing_terms(this->solution_u.size());
    this->assemble_forcing_terms(forcing_terms);

    this->system_rhs += forcing_terms;

    this->matrix_v.copy_from(this->mass_matrix);
  }
} // namespace Step23


int
main(int argc, char **argv)
{
  initlog();

  deallog.depth_file(1);

  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  try
    {
      Step23::Step23<2> wave_equation_solver;
      wave_equation_solver.run();
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
