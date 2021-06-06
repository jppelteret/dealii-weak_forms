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

#ifndef dealii_weakforms_assembler_matrix_based_h
#define dealii_weakforms_assembler_matrix_based_h

#include <deal.II/base/config.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>

#include <weak_forms/assembler_base.h>
#include <weak_forms/config.h>



WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  template <int dim,
            int spacedim           = dim,
            typename ScalarType    = double,
            bool use_vectorization = internal::UseVectorization::value>
  class MatrixBasedAssembler
    : public AssemblerBase<dim, spacedim, ScalarType, use_vectorization>
  {
    template <typename CellIteratorType,
              typename ScratchData,
              typename CopyData>
    using CellWorkerType =
      std::function<void(const CellIteratorType &, ScratchData &, CopyData &)>;

    template <typename CellIteratorType,
              typename ScratchData,
              typename CopyData>
    using BoundaryWorkerType = std::function<void(const CellIteratorType &,
                                                  const unsigned int &,
                                                  ScratchData &,
                                                  CopyData &)>;

    template <typename CellIteratorType,
              typename ScratchData,
              typename CopyData>
    using FaceWorkerType = std::function<void(const CellIteratorType &,
                                              const unsigned int,
                                              const unsigned int,
                                              const CellIteratorType &,
                                              const unsigned int,
                                              const unsigned int,
                                              ScratchData &,
                                              CopyData &)>;

  public:
    explicit MatrixBasedAssembler()
      : AssemblerBase<dim, spacedim, ScalarType, use_vectorization>(){};

    /**
     * Assemble the linear system matrix, excluding boundary and internal
     * face contributions.
     *
     * @tparam ScalarType
     * @tparam MatrixType
     * @param system_matrix
     * @param constraints
     *
     * @note Does not reset the matrix, so one can assemble from multiple
     * Assemblers into one matrix. This is useful if you want different
     * quadrature rules for different contributions on the same cell.
     */
    template <typename MatrixType,
              typename DoFHandlerType,
              typename CellQuadratureType>
    void
    assemble_matrix(MatrixType &                         system_matrix,
                    const AffineConstraints<ScalarType> &constraints,
                    const DoFHandlerType &               dof_handler,
                    const CellQuadratureType &           cell_quadrature) const
    {
      do_assemble_system<MatrixType, std::nullptr_t, std::nullptr_t>(
        &system_matrix,
        nullptr /*system_vector*/,
        constraints,
        dof_handler,
        nullptr /*solution_vector*/,
        cell_quadrature,
        nullptr /*face_quadrature*/);
    }

    // Same as the previous function, but with a solution vector
    template <typename MatrixType,
              typename VectorType,
              typename DoFHandlerType,
              typename CellQuadratureType>
    void
    assemble_matrix(MatrixType &                         system_matrix,
                    const VectorType &                   solution_vector,
                    const AffineConstraints<ScalarType> &constraints,
                    const DoFHandlerType &               dof_handler,
                    const CellQuadratureType &           cell_quadrature) const
    {
      do_assemble_system<MatrixType, std::nullptr_t, std::nullptr_t>(
        &system_matrix,
        nullptr /*system_vector*/,
        constraints,
        dof_handler,
        &solution_vector,
        cell_quadrature,
        nullptr /*face_quadrature*/);
    }

    /**
     * Assemble the linear system matrix, including boundary and internal
     * face contributions.
     *
     * @tparam ScalarType
     * @tparam MatrixType
     * @param system_matrix
     * @param constraints
     *
     * @note Does not reset the matrix, so one can assemble from multiple
     * Assemblers into one matrix. This is useful if you want different
     * quadrature rules for different contributions on the same cell.
     */
    template <typename MatrixType,
              typename DoFHandlerType,
              typename CellQuadratureType,
              typename FaceQuadratureType>
    void
    assemble_matrix(MatrixType &                         system_matrix,
                    const AffineConstraints<ScalarType> &constraints,
                    const DoFHandlerType &               dof_handler,
                    const CellQuadratureType &           cell_quadrature,
                    const FaceQuadratureType &           face_quadrature) const
    {
      do_assemble_system<MatrixType, std::nullptr_t, FaceQuadratureType>(
        &system_matrix,
        nullptr /*system_vector*/,
        constraints,
        dof_handler,
        nullptr /*solution_vector*/,
        cell_quadrature,
        &face_quadrature);
    }

    // Same as the previous function, but with a solution vector
    template <typename MatrixType,
              typename VectorType,
              typename DoFHandlerType,
              typename CellQuadratureType,
              typename FaceQuadratureType>
    void
    assemble_matrix(MatrixType &                         system_matrix,
                    const VectorType &                   solution_vector,
                    const AffineConstraints<ScalarType> &constraints,
                    const DoFHandlerType &               dof_handler,
                    const CellQuadratureType &           cell_quadrature,
                    const FaceQuadratureType &           face_quadrature) const
    {
      do_assemble_system<MatrixType, std::nullptr_t, FaceQuadratureType>(
        &system_matrix,
        nullptr /*system_vector*/,
        constraints,
        dof_handler,
        &solution_vector,
        cell_quadrature,
        &face_quadrature);
    }

    /**
     * Assemble a RHS vector, boundary and internal face contributions.
     *
     * @tparam ScalarType
     * @tparam MatrixType
     * @param system_matrix
     * @param constraints
     *
     * @note Does not reset the matrix, so one can assemble from multiple
     * Assemblers into one matrix. This is useful if you want different
     * quadrature rules for different contributions on the same cell.
     */
    template <typename VectorType,
              typename DoFHandlerType,
              typename CellQuadratureType>
    void
    assemble_rhs_vector(VectorType &                         system_vector,
                        const AffineConstraints<ScalarType> &constraints,
                        const DoFHandlerType &               dof_handler,
                        const CellQuadratureType &cell_quadrature) const
    {
      do_assemble_system<std::nullptr_t, VectorType, std::nullptr_t>(
        nullptr /*system_matrix*/,
        &system_vector,
        constraints,
        dof_handler,
        nullptr /*solution_vector*/,
        cell_quadrature,
        nullptr /*face_quadrature*/);
    }

    // Same as the previous function, but with a solution vector
    template <typename VectorType,
              typename DoFHandlerType,
              typename CellQuadratureType>
    void
    assemble_rhs_vector(VectorType &                         system_vector,
                        const VectorType &                   solution_vector,
                        const AffineConstraints<ScalarType> &constraints,
                        const DoFHandlerType &               dof_handler,
                        const CellQuadratureType &cell_quadrature) const
    {
      do_assemble_system<std::nullptr_t, VectorType, std::nullptr_t>(
        nullptr /*system_matrix*/,
        &system_vector,
        constraints,
        dof_handler,
        &solution_vector,
        cell_quadrature,
        nullptr /*face_quadrature*/);
    }

    /**
     * Assemble a RHS vector, including boundary and internal face
     * contributions.
     *
     * @tparam ScalarType
     * @tparam MatrixType
     * @param system_matrix
     * @param constraints
     *
     * @note Does not reset the matrix, so one can assemble from multiple
     * Assemblers into one matrix. This is useful if you want different
     * quadrature rules for different contributions on the same cell.
     */
    template <typename VectorType,
              typename DoFHandlerType,
              typename CellQuadratureType,
              typename FaceQuadratureType>
    void
    assemble_rhs_vector(VectorType &                         system_vector,
                        const AffineConstraints<ScalarType> &constraints,
                        const DoFHandlerType &               dof_handler,
                        const CellQuadratureType &           cell_quadrature,
                        const FaceQuadratureType &face_quadrature) const
    {
      do_assemble_system<std::nullptr_t, VectorType, FaceQuadratureType>(
        nullptr /*system_matrix*/,
        &system_vector,
        constraints,
        dof_handler,
        nullptr /*solution_vector*/,
        cell_quadrature,
        &face_quadrature);
    }

    // Same as the previous function, but with a solution vector
    template <typename VectorType,
              typename DoFHandlerType,
              typename CellQuadratureType,
              typename FaceQuadratureType>
    void
    assemble_rhs_vector(VectorType &                         system_vector,
                        const VectorType &                   solution_vector,
                        const AffineConstraints<ScalarType> &constraints,
                        const DoFHandlerType &               dof_handler,
                        const CellQuadratureType &           cell_quadrature,
                        const FaceQuadratureType &face_quadrature) const
    {
      do_assemble_system<std::nullptr_t, VectorType, FaceQuadratureType>(
        nullptr /*system_matrix*/,
        &system_vector,
        constraints,
        dof_handler,
        &solution_vector,
        cell_quadrature,
        &face_quadrature);
    }

    /**
     * Assemble a system matrix and a RHS vector, excluding boundary and
     * internal face contributions.
     *
     * @tparam ScalarType
     * @tparam MatrixType
     * @param system_matrix
     * @param constraints
     *
     * @note Does not reset the matrix, so one can assemble from multiple
     * Assemblers into one matrix. This is useful if you want different
     * quadrature rules for different contributions on the same cell.
     */
    template <typename MatrixType,
              typename VectorType,
              typename DoFHandlerType,
              typename CellQuadratureType>
    void
    assemble_system(MatrixType &                         system_matrix,
                    VectorType &                         system_vector,
                    const AffineConstraints<ScalarType> &constraints,
                    const DoFHandlerType &               dof_handler,
                    const CellQuadratureType &           cell_quadrature) const
    {
      do_assemble_system<MatrixType, VectorType, std::nullptr_t>(
        &system_matrix,
        &system_vector,
        constraints,
        dof_handler,
        nullptr /*solution_vector*/,
        cell_quadrature,
        nullptr /*face_quadrature*/);
    }

    // Same as the previous function, but with a solution vector
    template <typename MatrixType,
              typename VectorType,
              typename DoFHandlerType,
              typename CellQuadratureType>
    void
    assemble_system(MatrixType &                         system_matrix,
                    VectorType &                         system_vector,
                    const VectorType &                   solution_vector,
                    const AffineConstraints<ScalarType> &constraints,
                    const DoFHandlerType &               dof_handler,
                    const CellQuadratureType &           cell_quadrature) const
    {
      do_assemble_system<MatrixType, VectorType, std::nullptr_t>(
        &system_matrix,
        &system_vector,
        constraints,
        dof_handler,
        &solution_vector,
        cell_quadrature,
        nullptr /*face_quadrature*/);
    }

    /**
     * Assemble a system matrix and a RHS vector, including boundary and
     * internal face contributions.
     *
     * @tparam ScalarType
     * @tparam MatrixType
     * @param system_matrix
     * @param constraints
     *
     * @note Does not reset the matrix, so one can assemble from multiple
     * Assemblers into one matrix. This is useful if you want different
     * quadrature rules for different contributions on the same cell.
     */
    template <typename MatrixType,
              typename VectorType,
              typename DoFHandlerType,
              typename CellQuadratureType,
              typename FaceQuadratureType>
    void
    assemble_system(MatrixType &                         system_matrix,
                    VectorType &                         system_vector,
                    const AffineConstraints<ScalarType> &constraints,
                    const DoFHandlerType &               dof_handler,
                    const CellQuadratureType &           cell_quadrature,
                    const FaceQuadratureType &           face_quadrature) const
    {
      do_assemble_system<MatrixType, VectorType, FaceQuadratureType>(
        &system_matrix,
        &system_vector,
        constraints,
        dof_handler,
        nullptr /*solution_vector*/,
        cell_quadrature,
        &face_quadrature);
    }

    // Same as the previous function, but with a solution vector
    template <typename MatrixType,
              typename VectorType,
              typename DoFHandlerType,
              typename CellQuadratureType,
              typename FaceQuadratureType>
    void
    assemble_system(MatrixType &                         system_matrix,
                    VectorType &                         system_vector,
                    const VectorType &                   solution_vector,
                    const AffineConstraints<ScalarType> &constraints,
                    const DoFHandlerType &               dof_handler,
                    const CellQuadratureType &           cell_quadrature,
                    const FaceQuadratureType &           face_quadrature) const
    {
      do_assemble_system<MatrixType, VectorType, FaceQuadratureType>(
        &system_matrix,
        &system_vector,
        constraints,
        dof_handler,
        &solution_vector,
        cell_quadrature,
        &face_quadrature);
    }


  private:
    // TODO: ScratchData supports face quadrature without cell quadrature.
    //       But does mesh loop? Check this out...
    template <typename MatrixType,
              typename VectorType,
              typename FaceQuadratureType,
              typename DoFHandlerType,
              typename CellQuadratureType>
    void
    do_assemble_system(
      MatrixType *const                                system_matrix,
      VectorType *const                                system_vector,
      const AffineConstraints<ScalarType> &            constraints,
      const DoFHandlerType &                           dof_handler,
      const typename identity<VectorType>::type *const solution_vector,
      const CellQuadratureType &                       cell_quadrature,
      const FaceQuadratureType *const                  face_quadrature) const
    {
      using SolutionStorage_t =
        SolutionStorage<typename identity<VectorType>::type>;

      // We can only initialize the solution storage if the input
      // vector points to something valid.
      const SolutionStorage_t solution_storage(
        solution_vector != nullptr ? SolutionStorage_t(*solution_vector) :
                                     SolutionStorage_t());

      do_assemble_system<MatrixType,
                         VectorType,
                         FaceQuadratureType,
                         DoFHandlerType,
                         CellQuadratureType>(system_matrix,
                                             system_vector,
                                             constraints,
                                             dof_handler,
                                             solution_storage,
                                             cell_quadrature,
                                             face_quadrature);
    }


    template <typename MatrixType,
              typename VectorType,
              typename FaceQuadratureType,
              typename DoFHandlerType,
              typename CellQuadratureType>
    void
    do_assemble_system(
      MatrixType *const                    system_matrix,
      VectorType *const                    system_vector,
      const AffineConstraints<ScalarType> &constraints,
      const DoFHandlerType &               dof_handler,
      const SolutionStorage<typename identity<VectorType>::type>
        &                             solution_storage,
      const CellQuadratureType &      cell_quadrature,
      const FaceQuadratureType *const face_quadrature) const
    {
      static_assert(DoFHandlerType::dimension == dim,
                    "Dimension is incompatible");
      static_assert(DoFHandlerType::space_dimension == spacedim,
                    "Space dimension is incompatible");

      Assert(system_matrix || system_vector,
             ExcMessage("Either the system matrix or system RHS vector have "
                        "to be supplied in order for assembly to occur."));

      // if (!cell_quadrature)
      //   Assert(this->cell_vector_operations.empty(),
      //         ExcMessage("Assembly with no cell quadrature has been selected,
      //         "
      //                     "while there are boundary face contributions in to
      //                     the " "linear form. You should use the other
      //                     assemble_rhs_vector() " "function that takes in
      //                     cell quadrature as an argument so " "that all
      //                     contributions are considered."));

      if (!face_quadrature)
        {
          if (system_matrix)
            {
              Assert(
                this->boundary_face_matrix_operations.empty(),
                ExcMessage(
                  "Assembly with no face quadrature has been selected, "
                  "while there are boundary face contributions in to the "
                  "bilinear form. You should use the other assemble_matrix() "
                  "function that takes in face quadrature as an argument so "
                  "that all contributions are considered."));

              Assert(
                this->interface_face_matrix_operations.empty(),
                ExcMessage(
                  "Assembly with no face quadrature has been selected, "
                  "while there are internal face contributions in to the "
                  "bilinear form. You should use the other assemble_matrix() "
                  "function that takes in face quadrature as an argument so "
                  "that all contributions are considered."));
            }
          if (system_vector)
            {
              Assert(
                this->boundary_face_vector_operations.empty(),
                ExcMessage(
                  "Assembly with no face quadrature has been selected, "
                  "while there are boundary face contributions in to the "
                  "linear form. You should use the other assemble_rhs_vector() "
                  "function that takes in face quadrature as an argument so "
                  "that all contributions are considered."));
              Assert(
                this->interface_face_vector_operations.empty(),
                ExcMessage(
                  "Assembly with no interface quadrature has been selected, "
                  "while there are internal face contributions in to the "
                  "linear form. You should use the other assemble_rhs_vector() "
                  "function that takes in face quadrature as an argument so "
                  "that all contributions are considered."));
            }
        }

      using CellIteratorType = typename DoFHandlerType::active_cell_iterator;
      using ScratchData      = MeshWorker::ScratchData<dim, spacedim>;
      using CopyData         = MeshWorker::CopyData<1, 1, 1>;

      // Define a cell worker
      const auto &cell_matrix_operations = this->cell_matrix_operations;
      const auto &cell_vector_operations = this->cell_vector_operations;
      const auto &cell_field_solution_operations =
        this->cell_field_solution_operations;
      const auto &cell_ad_sd_operations = this->cell_ad_sd_operations;

      auto cell_worker =
        CellWorkerType<CellIteratorType, ScratchData, CopyData>();
      if (!cell_matrix_operations.empty() || !cell_vector_operations.empty())
        {
          cell_worker = [&cell_matrix_operations,
                         &cell_vector_operations,
                         &cell_field_solution_operations,
                         &cell_ad_sd_operations,
                         system_matrix,
                         system_vector,
                         solution_storage](const CellIteratorType &cell,
                                           ScratchData &           scratch_data,
                                           CopyData &              copy_data) {
            const auto &fe_values = scratch_data.reinit(cell);
            copy_data             = CopyData(fe_values.dofs_per_cell);
            copy_data.local_dof_indices[0] =
              scratch_data.get_local_dof_indices();

            // Extract the local solution vector, if it has been provided by the
            // user.
            if (solution_storage.n_solution_vectors() > 0)
              internal::extract_solution_local_dof_values(scratch_data,
                                                          solution_storage);

            // TODO: Is this actually required? Don't the functors cache as we
            // go along? Or is it user defined functions that I'm thinking
            // about... First we cache all field solutions into the
            // scratch_data. We do this because some of the user-defined
            // functors that use the cache might expect that these and other
            // solution fields already exist in the cache.
            for (const auto &cell_field_solution_op :
                 cell_field_solution_operations)
              cell_field_solution_op(scratch_data,
                                     solution_storage.get_solution_names());

            // Next we perform all operations that use AD or SD functors.
            // Although the forms are self-linearizing, they reference the
            // ADHelpers or SD BatchOptimizers that are stored in the form. So
            // these need to be updated with this cell/QP data before their
            // associated self-linearized forms, which require this data,
            // can be invoked.
            if (cell_ad_sd_operations.size() > 0)
              {
                Assert(
                  solution_storage.n_solution_vectors() > 0,
                  ExcMessage(
                    "The solution vector is required in order to perform "
                    "computations using automatic or symbolic differentiation."));
              }
            for (const auto &cell_ad_sd_op : cell_ad_sd_operations)
              cell_ad_sd_op(scratch_data,
                            solution_storage.get_solution_names());

            // Perform all operations that contribute to the local cell matrix
            if (system_matrix)
              {
                FullMatrix<ScalarType> &cell_matrix = copy_data.matrices[0];
                for (const auto &cell_matrix_op : cell_matrix_operations)
                  {
                    // We pass in solution_storage.get_solution_names() here
                    // to decouple the VectorType that underlies SolutionStorage
                    // from the operation.
                    cell_matrix_op(cell_matrix,
                                   scratch_data,
                                   solution_storage.get_solution_names(),
                                   fe_values);
                  }
              }

            // Perform all operations that contribute to the local cell vector
            if (system_vector)
              {
                Vector<ScalarType> &cell_vector = copy_data.vectors[0];
                for (const auto &cell_vector_op : cell_vector_operations)
                  {
                    cell_vector_op(cell_vector,
                                   scratch_data,
                                   solution_storage.get_solution_names(),
                                   fe_values);
                  }
              }
          };
        }

      // Define a boundary worker
      const auto &boundary_face_matrix_operations =
        this->boundary_face_matrix_operations;
      const auto &boundary_face_vector_operations =
        this->boundary_face_vector_operations;
      const auto &boundary_face_field_solution_operations =
        this->boundary_face_field_solution_operations;
      const auto &boundary_face_ad_sd_operations =
        this->boundary_face_ad_sd_operations;

      auto boundary_worker =
        BoundaryWorkerType<CellIteratorType, ScratchData, CopyData>();
      if (!boundary_face_matrix_operations.empty() ||
          !boundary_face_vector_operations.empty())
        {
          boundary_worker = [&boundary_face_matrix_operations,
                             &boundary_face_vector_operations,
                             &boundary_face_field_solution_operations,
                             &boundary_face_ad_sd_operations,
                             system_matrix,
                             system_vector,
                             solution_storage](const CellIteratorType &cell,
                                               const unsigned int      face,
                                               ScratchData &scratch_data,
                                               CopyData &   copy_data) {
            Assert((cell->face(face)->at_boundary()),
                   ExcMessage("Cell face is not at the boundary."));

            const auto &fe_values      = scratch_data.reinit(cell);
            const auto &fe_face_values = scratch_data.reinit(cell, face);
            // Not permitted inside a boundary or face worker!
            // copy_data             = CopyData(fe_values.dofs_per_cell);
            copy_data.local_dof_indices[0] =
              scratch_data.get_local_dof_indices();

            // Extract the local solution vector, if it's provided.
            if (solution_storage.n_solution_vectors() > 0)
              internal::extract_solution_local_dof_values(scratch_data,
                                                          solution_storage);

            // TODO: Is this actually required? Don't the functors cache as we
            // go along? Or is it user defined functions that I'm thinking
            // about... First we cache all field solutions into the
            // scratch_data. We do this because some of the user-defined
            // functors that use the cache might expect that these and other
            // solution fields already exist in the cache.
            for (const auto &boundary_face_field_solution_op :
                 boundary_face_field_solution_operations)
              boundary_face_field_solution_op(
                scratch_data, solution_storage.get_solution_names());

            // Next we perform all operations that use AD or SD functors.
            // Although the forms are self-linearizing, they reference the
            // ADHelpers or SD BatchOptimizers that are stored in the form. So
            // these need to be updated with this cell/QP data before their
            // associated self-linearized forms, which require this data,
            // can be invoked.
            for (const auto &boundary_face_ad_sd_op :
                 boundary_face_ad_sd_operations)
              boundary_face_ad_sd_op(scratch_data,
                                     solution_storage.get_solution_names());

            // Perform all operations that contribute to the local cell matrix
            if (system_matrix)
              {
                FullMatrix<ScalarType> &cell_matrix = copy_data.matrices[0];
                for (const auto &boundary_face_matrix_op :
                     boundary_face_matrix_operations)
                  {
                    boundary_face_matrix_op(
                      cell_matrix,
                      scratch_data,
                      solution_storage.get_solution_names(),
                      fe_values,
                      fe_face_values,
                      face);
                  }
              }

            // Perform all operations that contribute to the local cell vector
            if (system_vector)
              {
                Vector<ScalarType> &cell_vector = copy_data.vectors[0];
                for (const auto &boundary_face_vector_op :
                     boundary_face_vector_operations)
                  {
                    boundary_face_vector_op(
                      cell_vector,
                      scratch_data,
                      solution_storage.get_solution_names(),
                      fe_values,
                      fe_face_values,
                      face);
                  }
              }
          };
        }

      // Define a face / interface worker
      const auto &interface_face_matrix_operations =
        this->interface_face_matrix_operations;
      const auto &interface_face_vector_operations =
        this->interface_face_vector_operations;
      auto face_worker =
        FaceWorkerType<CellIteratorType, ScratchData, CopyData>();
      if (!interface_face_matrix_operations.empty() ||
          !interface_face_vector_operations.empty())
        {
          // interface_vector_operations
          AssertThrow(
            false,
            ExcMessage(
              "Internal face cell matrix/vector contributions have not yet been implemented."));
        }

      // Symmetry of the global system
      const bool &global_system_symmetry_flag =
        this->global_system_symmetry_flag;

      auto copier = [&constraints,
                     system_matrix,
                     system_vector,
                     &global_system_symmetry_flag](const CopyData &copy_data) {
        const FullMatrix<ScalarType> &cell_matrix = copy_data.matrices[0];
        const Vector<ScalarType> &    cell_vector = copy_data.vectors[0];
        const std::vector<dealii::types::global_dof_index> &local_dof_indices =
          copy_data.local_dof_indices[0];

        // Copy the upper half (i.e. contributions below the diagonal) into the
        // lower half if the global system is marked as symmetric.
        if (global_system_symmetry_flag == true)
          {
            // Hmm... a bit nasty, but it makes sense to do the global
            // symmetrization only once if possible. To (unnecessarily)
            // symmetrize each form contribution after assembling only it's
            // lower diagonal part would be a little wasteful.
            FullMatrix<ScalarType> &symmetrized_cell_matrix =
              const_cast<FullMatrix<ScalarType> &>(cell_matrix);
            // symmetrized_cell_matrix.symmetrize();

            using DoFRange_t =
              std_cxx20::ranges::iota_view<unsigned int, unsigned int>;
            const DoFRange_t dof_range_j(0, local_dof_indices.size());
            for (const auto j : dof_range_j)
              {
                const DoFRange_t dof_range_i(j + 1, local_dof_indices.size());
                for (const auto i : dof_range_i)
                  symmetrized_cell_matrix(i, j) = cell_matrix(j, i);
              }
          }

        if (system_matrix && system_vector)
          {
            internal::distribute_local_to_global(constraints,
                                                 cell_matrix,
                                                 cell_vector,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_vector);
          }
        else if (system_matrix)
          {
            internal::distribute_local_to_global(constraints,
                                                 cell_matrix,
                                                 local_dof_indices,
                                                 system_matrix);
          }
        else if (system_vector)
          {
            internal::distribute_local_to_global(constraints,
                                                 cell_vector,
                                                 local_dof_indices,
                                                 system_vector);
          }
        else
          {
            AssertThrow(system_matrix || system_vector,
                        ExcMessage(
                          "Either the system matrix or system RHS vector have "
                          "to be supplied in order for assembly to occur."));
          }
      };

      // Initialize the assistant objects used during assembly.
      const ScratchData sample_scratch_data =
        (face_quadrature ?
           internal::construct_scratch_data<ScratchData, FaceQuadratureType>(
             dof_handler.get_fe(),
             cell_quadrature,
             this->get_cell_update_flags(),
             face_quadrature,
             this->get_face_update_flags()) :
           ScratchData(dof_handler.get_fe(),
                       cell_quadrature,
                       this->get_cell_update_flags()));
      const CopyData sample_copy_data(dof_handler.get_fe().dofs_per_cell);

      // Set the assembly flags, based off of the operations that we intend to
      // do.
      MeshWorker::AssembleFlags assembly_flags = MeshWorker::assemble_nothing;
      if (!cell_matrix_operations.empty() || !cell_vector_operations.empty())
        assembly_flags |= MeshWorker::assemble_own_cells;
      if (!boundary_face_matrix_operations.empty() ||
          !boundary_face_vector_operations.empty())
        assembly_flags |= MeshWorker::assemble_boundary_faces;
      if (!interface_face_matrix_operations.empty() ||
          !interface_face_vector_operations.empty())
        assembly_flags |= MeshWorker::assemble_own_interior_faces_once;

      // Finally! We can perform the assembly.
      if (assembly_flags)
        {
          MeshWorker::mesh_loop(dof_handler.active_cell_iterators(),
                                cell_worker,
                                copier,
                                sample_scratch_data,
                                sample_copy_data,
                                assembly_flags,
                                boundary_worker,
                                face_worker);

          if (system_matrix)
            {
              if (!cell_matrix_operations.empty() ||
                  !boundary_face_matrix_operations.empty() ||
                  !interface_face_matrix_operations.empty())
                {
                  internal::compress(system_matrix);
                }
            }

          if (system_vector)
            {
              if (!cell_vector_operations.empty() ||
                  !boundary_face_vector_operations.empty() ||
                  !interface_face_vector_operations.empty())
                {
                  internal::compress(system_vector);
                }
            }
        }
    }
  };

} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_assembler_matrix_based_h
