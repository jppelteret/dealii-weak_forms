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

#ifndef dealii_weakforms_space_extractors_h
#define dealii_weakforms_space_extractors_h

#include <deal.II/base/config.h>

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_values_extractors.h>

#include <weak_forms/config.h>
#include <weak_forms/numbers.h>
#include <weak_forms/types.h>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  /**
   * @brief A namespace dedicated to objects that help to identify subspaces within a global finite element space.
   */
  namespace SubSpaceExtractors
  {
    /**
     * @brief An object that can be used to extract a scalar subspace from a global finite element space.
     */
    struct Scalar
    {
      static const int rank = 0;

      Scalar(const types::field_index          field_index,
             const FEValuesExtractors::Scalar &extractor,
             const std::string &               field_ascii,
             const std::string &               field_latex)
        : extractor(extractor)
        , field_index(field_index)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      Scalar(const FEValuesExtractors::Scalar &extractor,
             const std::string &               field_ascii,
             const std::string &               field_latex)
        : Scalar(numbers::invalid_field_index,
                 extractor,
                 field_ascii,
                 field_latex)
      {}

      Scalar(const types::field_index field_index,
             const unsigned int       component,
             const std::string &      field_ascii,
             const std::string &      field_latex)
        : extractor(component)
        , field_index(field_index)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      Scalar(const unsigned int component,
             const std::string &field_ascii,
             const std::string &field_latex)
        : Scalar(numbers::invalid_field_index,
                 component,
                 field_ascii,
                 field_latex)
      {}

      const FEValuesExtractors::Scalar extractor;
      const types::field_index         field_index;
      const std::string                field_ascii;
      const std::string                field_latex;
    };



    /**
     * @brief An object that can be used to extract a vector subspace from a global finite element space.
     *
     * The number of vector components is determined on the basis of the space
     * from which the extraction is to be performed.
     */
    struct Vector
    {
      static const int rank = 1;

      Vector(const types::field_index          field_index,
             const FEValuesExtractors::Vector &extractor,
             const std::string &               field_ascii,
             const std::string &               field_latex)
        : extractor(extractor)
        , field_index(field_index)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      Vector(const FEValuesExtractors::Vector &extractor,
             const std::string &               field_ascii,
             const std::string &               field_latex)
        : Vector(numbers::invalid_field_index,
                 extractor,
                 field_ascii,
                 field_latex)
      {}

      Vector(const types::field_index field_index,
             const unsigned int       first_component,
             const std::string &      field_ascii,
             const std::string &      field_latex)
        : extractor(first_component)
        , field_index(field_index)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      Vector(const unsigned int first_component,
             const std::string &field_ascii,
             const std::string &field_latex)
        : Vector(numbers::invalid_field_index,
                 first_component,
                 field_ascii,
                 field_latex)
      {}

      const FEValuesExtractors::Vector extractor;
      const types::field_index         field_index;
      const std::string                field_ascii;
      const std::string                field_latex;
    };



    /**
     * @brief An object that can be used to extract a tensor subspace from a global finite element space.
     *
     * The number of tensor components is determined on the basis of the space
     * from which the extraction is to be performed.
     *
     * @tparam rank_ The rank of the tensor that this subspace is associated with.
     */
    template <int rank_>
    struct Tensor
    {
      static const int rank = rank_;

      Tensor(const types::field_index                field_index,
             const FEValuesExtractors::Tensor<rank> &extractor,
             const std::string &                     field_ascii,
             const std::string &                     field_latex)
        : extractor(extractor)
        , field_index(field_index)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      Tensor(const FEValuesExtractors::Tensor<rank> &extractor,
             const std::string &                     field_ascii,
             const std::string &                     field_latex)
        : Tensor(numbers::invalid_field_index,
                 extractor,
                 field_ascii,
                 field_latex)
      {}

      Tensor(const types::field_index field_index,
             const unsigned int       first_component,
             const std::string &      field_ascii,
             const std::string &      field_latex)
        : extractor(first_component)
        , field_index(field_index)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      Tensor(const unsigned int first_component,
             const std::string &field_ascii,
             const std::string &field_latex)
        : Tensor(numbers::invalid_field_index,
                 first_component,
                 field_ascii,
                 field_latex)
      {}

      const FEValuesExtractors::Tensor<rank> extractor;
      const types::field_index               field_index;
      const std::string                      field_ascii;
      const std::string                      field_latex;
    };


    /**
     * @brief An object that can be used to extract a symmetric tensor subspace from a global finite element space.
     *
     * The number of tensor components is determined on the basis of the space
     * from which the extraction is to be performed.
     *
     * @tparam rank_ The rank of the tensor that this subspace is associated with.
     */
    template <int rank_>
    struct SymmetricTensor
    {
      static const int rank = rank_;

      SymmetricTensor(
        const types::field_index                         field_index,
        const FEValuesExtractors::SymmetricTensor<rank> &extractor,
        const std::string &                              field_ascii,
        const std::string &                              field_latex)
        : extractor(extractor)
        , field_index(field_index)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      SymmetricTensor(
        const FEValuesExtractors::SymmetricTensor<rank> &extractor,
        const std::string &                              field_ascii,
        const std::string &                              field_latex)
        : SymmetricTensor(numbers::invalid_field_index,
                          extractor,
                          field_ascii,
                          field_latex)
      {}

      SymmetricTensor(const types::field_index field_index,
                      const unsigned int       first_component,
                      const std::string &      field_ascii,
                      const std::string &      field_latex)
        : extractor(first_component)
        , field_index(field_index)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      SymmetricTensor(const unsigned int first_component,
                      const std::string &field_ascii,
                      const std::string &field_latex)
        : SymmetricTensor(numbers::invalid_field_index,
                          first_component,
                          field_ascii,
                          field_latex)
      {}

      const FEValuesExtractors::SymmetricTensor<rank> extractor;
      const types::field_index                        field_index;
      const std::string                               field_ascii;
      const std::string                               field_latex;
    };
  } // namespace SubSpaceExtractors


  namespace internal
  {
    /**
     * @brief A helper class that identifies the association between a WeakForm subspace extractor and a deal.II extractor.
     *
     * @tparam FEValuesExtractors_t One of the extractors from the deal.II FEValuesExtractors namespace.
     */
    template <typename FEValuesExtractors_t>
    struct SubSpaceExtractor;



    /**
     * @brief A specialisation of the SubSpaceExtractor class for the extraction of scalar subspaces.
     */
    template <>
    struct SubSpaceExtractor<FEValuesExtractors::Scalar>
    {
      using type = WeakForms::SubSpaceExtractors::Scalar;
    };



    /**
     * @brief A specialisation of the SubSpaceExtractor class for the extraction of vector subspaces.
     */
    template <>
    struct SubSpaceExtractor<FEValuesExtractors::Vector>
    {
      using type = WeakForms::SubSpaceExtractors::Vector;
    };



    /**
     * @brief A specialisation of the SubSpaceExtractor class for the extraction of tensor subspaces.
     *
     * @tparam rank_ The rank of the tensor that this subspace is associated with.
     */
    template <int rank>
    struct SubSpaceExtractor<FEValuesExtractors::Tensor<rank>>
    {
      using type = WeakForms::SubSpaceExtractors::Tensor<rank>;
    };



    /**
     * @brief A specialisation of the SubSpaceExtractor class for the extraction of symmetric tensor subspaces.
     *
     * @tparam rank_ The rank of the tensor that this subspace is associated with.
     */
    template <int rank>
    struct SubSpaceExtractor<FEValuesExtractors::SymmetricTensor<rank>>
    {
      using type = WeakForms::SubSpaceExtractors::SymmetricTensor<rank>;
    };
  } // namespace internal

} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_space_extractors_h
