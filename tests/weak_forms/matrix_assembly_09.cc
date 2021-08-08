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


// Check that contractions of symmetric and non-symmetric tensors
// are permissible.
//
// Observed while putting together test step-44-variant_01e

#include <deal.II/physics/elasticity/standard_tensors.h>

#include <weak_forms/weak_forms.h>

#include "../weak_forms_tests.h"


template <int dim, int spacedim = dim>
void
run()
{
  using namespace WeakForms;

  LogStream::Prefix prefix("Dim " + dealii::Utilities::to_string(dim));

  // Symbolic types for test function, trial solution and a coefficient.
  const TestFunction<dim, spacedim>  test;
  const TrialSolution<dim, spacedim> trial;
  const FieldSolution<dim, spacedim> field_solution;
  const SubSpaceExtractors::Vector subspace_extractor_u(0, "u", "\\mathbf{u}");

  // Test function (subspaced)
  const auto grad_test_u = test[subspace_extractor_u].gradient();

  // Trial solution (subspaces)
  const auto grad_trial_u = trial[subspace_extractor_u].gradient();

  // Field solution
  const auto grad_u = field_solution[subspace_extractor_u].gradient();

  // Field variables
  const SymmetricTensorFunctor<2, spacedim> I_symb(
    "I", "\\mathbf{I}"); // Identity tensor
  const auto I = I_symb.template value<double, dim>(
    [](const FEValuesBase<dim, spacedim> &fe_values, const unsigned int)
    { return Physics::Elasticity::StandardTensors<dim>::I; });
  const SymmetricTensorFunctor<4, spacedim> H_symb(
    "H", "\\mathcal{H}"); // Constitutive tensor
  const auto H = H_symb.template value<double, dim>(
    [](const FEValuesBase<dim, spacedim> &fe_values, const unsigned int)
    { return SymmetricTensor<4, dim>(); });
  const TensorFunctor<4, spacedim> H_ns_symb(
    "HH", "\\mathcal{HH}"); // Constitutive tensor
  const auto H_ns = H_ns_symb.template value<double, dim>(
    [](const FEValuesBase<dim, spacedim> &fe_values, const unsigned int)
    { return Tensor<4, dim>(); });

  // Variations and linearisations
  const auto F     = I + grad_u;
  const auto dF    = grad_test_u;
  const auto DF    = grad_trial_u;
  const auto dE    = symmetrize(transpose(F) * dF);
  const auto DE    = symmetrize(transpose(F) * DF);
  const auto dE_ns = transpose(F) * dF;
  const auto DE_ns = transpose(F) * DF;

  // Non-vectorized assembler
  {
    constexpr bool use_vectorization = false;
    MatrixBasedAssembler<dim, spacedim, double, use_vectorization> assembler;

    // Rank 2 tensor . Scalar . Rank 2 tensor
    assembler += bilinear_form(dE, 1.0, DE).dV();       // Fully symmetric
    assembler += bilinear_form(dE_ns, 1.0, DE_ns).dV(); // Fully non-symmetric
    assembler += bilinear_form(dE_ns, 1.0, DE).dV();    // Mixed 1
    assembler += bilinear_form(dE, 1.0, DE_ns).dV();    // Mixed 2

    // Rank 2 tensor : Symmetric Rank 4 tensor : Rank 2 tensor
    assembler += bilinear_form(dE, H, DE).dV();       // Fully symmetric
    assembler += bilinear_form(dE_ns, H, DE_ns).dV(); // Fully non-symmetric
    assembler += bilinear_form(dE_ns, H, DE).dV();    // Mixed 1
    assembler += bilinear_form(dE, H, DE_ns).dV();    // Mixed 2

    // Rank 2 tensor : Non-symmetric Rank 4 tensor : Rank 2 tensor
    assembler += bilinear_form(dE, H_ns, DE).dV();       // Fully symmetric
    assembler += bilinear_form(dE_ns, H_ns, DE_ns).dV(); // Fully non-symmetric
    assembler += bilinear_form(dE_ns, H_ns, DE).dV();    // Mixed 1
    assembler += bilinear_form(dE, H_ns, DE_ns).dV();    // Mixed 2
  }

  // Vectorized assembler
  {
    constexpr bool use_vectorization = true;
    MatrixBasedAssembler<dim, spacedim, double, use_vectorization> assembler;

    // Rank 2 tensor . Scalar . Rank 2 tensor
    assembler += bilinear_form(dE, 1.0, DE).dV();       // Fully symmetric
    assembler += bilinear_form(dE_ns, 1.0, DE_ns).dV(); // Fully non-symmetric
    assembler += bilinear_form(dE_ns, 1.0, DE).dV();    // Mixed 1
    assembler += bilinear_form(dE, 1.0, DE_ns).dV();    // Mixed 2

    // Rank 2 tensor : Symmetric Rank 4 tensor : Rank 2 tensor
    assembler += bilinear_form(dE, H, DE).dV();       // Fully symmetric
    assembler += bilinear_form(dE_ns, H, DE_ns).dV(); // Fully non-symmetric
    assembler += bilinear_form(dE_ns, H, DE).dV();    // Mixed 1
    assembler += bilinear_form(dE, H, DE_ns).dV();    // Mixed 2

    // Rank 2 tensor : Non-symmetric Rank 4 tensor : Rank 2 tensor
    assembler += bilinear_form(dE, H_ns, DE).dV();       // Fully symmetric
    assembler += bilinear_form(dE_ns, H_ns, DE_ns).dV(); // Fully non-symmetric
    assembler += bilinear_form(dE_ns, H_ns, DE).dV();    // Mixed 1
    assembler += bilinear_form(dE, H_ns, DE_ns).dV();    // Mixed 2
  }

  deallog << "OK" << std::endl;
}


int
main(int argc, char *argv[])
{
  initlog();
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  run<2>();
  run<3>();

  deallog << "OK" << std::endl;
}
